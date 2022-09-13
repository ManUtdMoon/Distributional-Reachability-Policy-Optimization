import copy
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from .config import BaseConfig, Configurable, Optional
from .defaults import ACTOR_LR, OPTIMIZER
from .log import default_log as log
from .policy import BasePolicy, SquashedGaussianPolicy
from .torch_util import device, Module, mlp, update_ema, freeze_module, torchify
from .util import pythonic_mean


class CriticEnsemble(Configurable, Module):
    class Config(BaseConfig):
        n_critics = 2
        hidden_layers = 2
        hidden_dim = 256

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), 1]
        self.qs = torch.nn.ModuleList([
            mlp(dims, squeeze_output=True) for _ in range(self.n_critics)
        ])

    def all(self, state, action):
        sa = torch.cat([state, action], -1)
        return [q(sa) for q in self.qs]

    def min(self, state, action):
        return torch.min(*self.all(state, action))

    def mean(self, state, action):
        return pythonic_mean(self.all(state, action))

    def random_choice(self, state, action):
        sa = torch.cat([state, action], -1)
        return random.choice(self.qs)(sa)


class SSAC(BasePolicy, Module):
    class Config(BaseConfig):
        discount = 0.99
        init_alpha = 1.0
        autotune_alpha = True
        target_entropy = Optional(float)
        use_log_alpha_loss = False
        deterministic_backup = False
        critic_update_multiplier = 1
        actor_lr = 8e-5
        actor_lr_end = 4e-5
        critic_lr = 3e-4
        critic_lr_end = 8e-5
        critic_cfg = CriticEnsemble.Config()
        tau = 0.005
        
        actor_update_interval = 2

        batch_size = 256
        hidden_dim = 256
        hidden_layers = 2
        update_violation_cost = False  # TODO: if set to False: SMBPO -> MBPO
        
        grad_norm = 5.

    def __init__(self, config, state_dim, action_dim, con_dim, 
                 horizon, epochs, steps_per_epoch, solver_updates_per_step,
                 env_factory, model_ensemble, optimizer_factory=OPTIMIZER):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.con_dim = con_dim
        self.horizon = horizon
        self.violation_cost = 0.0
        self.updates_per_training = epochs * steps_per_epoch * solver_updates_per_step
        self.actor_updates_num = int(self.updates_per_training / self.actor_update_interval)

        self.actor = SquashedGaussianPolicy(mlp(
            [state_dim, *([self.hidden_dim] * self.hidden_layers), action_dim*2]
        ))
        self.critic = CriticEnsemble(self.critic_cfg, state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        freeze_module(self.critic_target)

        self.critic_optimizer = optimizer_factory(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-4)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=self.updates_per_training,
            eta_min=self.critic_lr_end
        )

        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr, weight_decay=1e-4)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=self.actor_updates_num,
            eta_min=self.actor_lr_end
        )

        log_alpha = torch.tensor(math.log(self.init_alpha), device=device, requires_grad=True)
        self.log_alpha = log_alpha
        if self.autotune_alpha:
            self.alpha_optimizer = optimizer_factory([self.log_alpha], lr=self.actor_lr)
        if self.target_entropy is None:
            self.target_entropy = -action_dim   # set target entropy to -dim(A)

        self.criterion = nn.MSELoss()

        self.register_buffer('total_updates', torch.zeros([]))

    def act(self, states, eval):
        return self.actor.act(states, eval)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def violation_value(self):
        return -self.violation_cost / (1. - self.discount)

    def update_r_bounds(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        if self.update_violation_cost:
            self.violation_cost = (r_max - r_min) / self.discount**self.horizon - r_max
        log.message(f'r bounds: [{r_min, r_max}], C = {self.violation_cost}')

    # def critic_loss(self, obs, action, next_obs, reward, done):  # maybe useless
    #     reward = reward.clamp(self.r_min, self.r_max)
    #     target = super().compute_target(next_obs, reward, done)
    #     if done.any():
    #         target[done] = 0.  # self.terminal_value
    #     return self.critic_loss_given_target(obs, action, target)

    def compute_target(self, next_obs, reward, done, violation):
        with torch.no_grad():
            distr = self.actor.distr(next_obs)
            next_action = distr.sample()
            log_prob = distr.log_prob(next_action)
            next_value = self.critic_target.min(next_obs, next_action)
            if not self.deterministic_backup:
                next_value = next_value - self.alpha.detach() * log_prob
            q = reward + self.discount * (1. - done.float()) * next_value
            q[violation] = self.violation_value
            return q

    def critic_loss_given_target(self, obs, action, target):
        qs = self.critic.all(obs, action)
        return pythonic_mean([self.criterion(q, target) for q in qs])

    def critic_loss(self, obs, action, next_obs, reward, done, violation):
        target = self.compute_target(next_obs, reward, done, violation)
        return self.critic_loss_given_target(obs, action, target)

    def update_critic(self, *critic_loss_args):
        critic_loss = self.critic_loss(*critic_loss_args)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm)
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()
        update_ema(self.critic_target, self.critic, self.tau)
        return critic_loss.detach()

    def actor_loss(self, obs, include_alpha=True):
        distr = self.actor.distr(obs)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        actor_Q = self.critic.random_choice(obs, action)
        alpha = self.alpha
        actor_loss = torch.mean(alpha.detach() * log_prob - actor_Q)
        if include_alpha:
            multiplier = self.log_alpha if self.use_log_alpha_loss else alpha
            alpha_loss = -multiplier * torch.mean(log_prob.detach() + self.target_entropy)
            return [actor_loss, alpha_loss]
        else:
            return [actor_loss]

    def update_actor_and_alpha(self, obs):
        losses = self.actor_loss(obs, include_alpha=self.autotune_alpha)
        optimizers = [self.actor_optimizer, self.alpha_optimizer] if self.autotune_alpha else \
                     [self.actor_optimizer]
        assert len(losses) == len(optimizers)
        for i, (loss, optimizer) in enumerate(zip(losses, optimizers)):
            optimizer.zero_grad()
            loss.backward()
            if i == 0:  # for actor only: clip grad
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
            optimizer.step()
            if i == 0:  # for actor only: lr schedule
                self.actor_lr_scheduler.step()

    def update(self, replay_buffer):
        assert self.critic_update_multiplier >= 1
        for _ in range(self.critic_update_multiplier):
            samples = replay_buffer.sample(self.batch_size)
            self.update_critic(*samples)
        self.update_actor_and_alpha(samples[0])
        self.total_updates += 1