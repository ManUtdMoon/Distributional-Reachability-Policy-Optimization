import copy
import math
import random

import torch
from torch import nn

from .config import BaseConfig, Configurable, Optional
from .defaults import ACTOR_LR, OPTIMIZER
from .log import default_log as log
from .policy import BasePolicy, SquashedGaussianPolicy
from .torch_util import device, Module, mlp, update_ema, freeze_module
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
        sa = torch.cat([state, action], 1)
        return [q(sa) for q in self.qs]

    def min(self, state, action):
        return torch.min(*self.all(state, action))

    def mean(self, state, action):
        return pythonic_mean(self.all(state, action))

    def random_choice(self, state, action):
        sa = torch.cat([state, action], 1)
        return random.choice(self.qs)(sa)


class ConstraintCritic(Configurable, Module):
    class Config(BaseConfig):
        hidden_layers = 2
        hidden_dim = 256

    def __init__(self, config, state_dim, action_dim, output_dim, output_activation=None):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), output_dim]
        self.qc = mlp(dims, output_activation=output_activation, squeeze_output=True)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.qc(sa)


class MLPMultiplier(Configurable, Module):
    class Config(BaseConfig) :
        hidden_layers = 2
        hidden_dim = 256

    def __init__(self, config, state_dim, max_multiplier=100):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim+1, *([self.hidden_dim] * self.hidden_layers), 1]
        self.lam = mlp(dims, activation='tanh', output_activation='identity', squeeze_output=True)
        self.max_lam = max_multiplier
    
    def forward(self, state, Qc):  # TODO:state input, lambda_max as hyperparameter
        sa = torch.cat([state, Qc.unsqueeze(-1)], 1)
        lam = self.max_lam/2 + self.max_lam/2 * torch.tanh(self.lam(sa)/self.max_lam*2)
        return lam


class SSAC(BasePolicy, Module):
    class Config(BaseConfig):
        discount = 0.99
        init_alpha = 1.0
        autotune_alpha = True
        target_entropy = Optional(float)
        use_log_alpha_loss = False
        deterministic_backup = False

        critic_update_multiplier = 1
        actor_lr = ACTOR_LR
        actor_lr_end = 5e-5
        critic_lr = 3e-4
        critic_lr_end = 8e-5
        multiplier_lr = 3e-4
        multiplier_lr_end = 1e-5
        critic_cfg = CriticEnsemble.Config()
        constraint_critic_cfg = ConstraintCritic.Config()
        mlp_multiplier_cfg = MLPMultiplier.Config()
        tau = 0.005
        
        actor_update_interval = 2

        batch_size = 256
        hidden_dim = 256
        hidden_layers = 2
        update_violation_cost = False  # TODO: if set to False: SMBPO -> MBPO
        
        grad_norm = 5.

        # safety-related hyper-params
        constraint_threshold = 0.
        constrained_fcn = 'reachability'
        mlp_multiplier = True
        max_multiplier = 50.0
        penalty_lb = -1.0
        penalty_ub = 100.
        # penalty_offset = 1.0
        fixed_multiplier = 15.0
        multiplier_update_interval = 5

    def __init__(self, config, state_dim, action_dim, con_dim, horizon,
                 optimizer_factory=OPTIMIZER):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.con_dim = con_dim
        self.horizon = horizon
        self.violation_cost = 0.0
        # epochs * steps_per_epoch * solver_updates_per_step
        # because we cannot pass the higher config to here, so we put it here and it is super ugly. We admit it.
        self.updates_per_training = 200 * 360 * 10
        self.lam_updates_num = int(self.updates_per_training / self.multiplier_update_interval)
        self.actor_updates_num = int(self.updates_per_training / self.actor_update_interval)

        # -------- actor & critic (incl. constraint) -------- #
        self.actor = SquashedGaussianPolicy(mlp(
            [state_dim, *([self.hidden_dim] * self.hidden_layers), action_dim*2]
        ))
        self.actor_safe = copy.deepcopy(self.actor)
        self.critic = CriticEnsemble(self.critic_cfg, state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        freeze_module(self.critic_target)
        output_dim = self.con_dim if self.constrained_fcn == 'reachability' else 1
        self.constraint_critic = ConstraintCritic(
            self.constraint_critic_cfg, state_dim, action_dim, output_dim=output_dim,
            output_activation='softplus' if self.constrained_fcn == 'cost' else None
        )
        self.constraint_critic_target = copy.deepcopy(self.constraint_critic)
        freeze_module(self.constraint_critic_target)

        self.critic_optimizer = optimizer_factory(
            list(self.critic.parameters()) + list(self.constraint_critic.parameters()), 
            lr=self.critic_lr,
            weight_decay=1e-4
        )
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

        self.actor_safe_optimizer = optimizer_factory(self.actor_safe.parameters(), lr=self.actor_lr, weight_decay=1e-4)
        self.actor_safe_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_safe_optimizer,
            T_max=self.actor_updates_num,
            eta_min=self.actor_lr_end
        )

        # -------- alpha in SAC -------- #
        log_alpha = torch.tensor(math.log(self.init_alpha), device=device, requires_grad=True)
        self.log_alpha = log_alpha
        if self.autotune_alpha:
            self.alpha_optimizer = optimizer_factory([self.log_alpha], lr=self.actor_lr)
        if self.target_entropy is None:
            self.target_entropy = -action_dim   # set target entropy to -dim(A)

        # -------- multiplier for safety -------- #
        if self.mlp_multiplier:
            self.multiplier = MLPMultiplier(self.mlp_multiplier_cfg, state_dim, self.max_multiplier)
            self.multiplier_optimizer = optimizer_factory(self.multiplier.parameters(), lr=self.multiplier_lr, weight_decay=1e-4)
            self.multiplier_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.multiplier_optimizer,
                T_max=self.lam_updates_num,
                eta_min=self.multiplier_lr_end
            )
        else:
            self.multiplier = nn.parameter.Parameter(
                torch.tensor(10., device=device, dtype=torch.float)  # todo: a larger initial multiplier
            )
            self.multiplier_optimizer = optimizer_factory(
                [self.multiplier], 
                lr=self.multiplier_lr
            )

        self.criterion = nn.MSELoss()

        self.register_buffer('total_updates', torch.zeros([]))

    def act(self, states, eval):
        return self.actor.act(states, eval)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    @property
    def lam(self):
        assert not self.mlp_multiplier
        assert self.multiplier.shape == ()
        return nn.functional.softplus(self.multiplier)

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
            # q[violation] = self.violation_value  # TODO: if commented, SMBPO -> MBPO
            return q

    def critic_loss_given_target(self, obs, action, target):
        qs = self.critic.all(obs, action)
        return pythonic_mean([self.criterion(q, target) for q in qs])

    def critic_loss(self, obs, action, next_obs, reward, done, violation, constraint_value):
        target = self.compute_target(next_obs, reward, done, violation)
        return self.critic_loss_given_target(obs, action, target)
    
    def compute_cons_target(self, next_obs, done, violation, constraint_value):
        with torch.no_grad():
            if self.constrained_fcn == 'cost':
                distr = self.actor.distr(next_obs)
                next_action = distr.sample()
                next_qc_value = self.constraint_critic_target(next_obs, next_action)
                qc = violation.float() + self.discount * (1. - done.float()) * next_qc_value
            elif self.constrained_fcn == 'reachability':
                distr = self.actor_safe.distr(next_obs)
                next_action = distr.sample()
                next_qc_value = self.constraint_critic_target(next_obs, next_action)
                qc_nonterminal = (1. - self.discount) * constraint_value.float() + self.discount * torch.maximum(constraint_value.float(), next_qc_value)
                dones = done.tile((self.con_dim, 1)).t().float()
                qc = qc_nonterminal * (1 - dones.float()) + constraint_value * dones.float()
                assert qc.shape == qc_nonterminal.shape
            else:
                raise NotImplementedError
            return qc

    def cons_critic_loss_given_target(self, obs, action, target):
        qcs = self.constraint_critic(obs, action)
        return self.criterion(qcs, target)

    def constraint_critic_loss(self, obs, action, next_obs, reward, done, violation, constraint_value):
        target = self.compute_cons_target(next_obs, done, violation, constraint_value)
        return self.cons_critic_loss_given_target(obs, action, target)

    def update_critic(self, *critic_loss_args):
        self.critic_optimizer.zero_grad()

        # critic part
        critic_loss = self.critic_loss(*critic_loss_args)
        
        # constraint_critic part
        constraint_critic_loss = self.constraint_critic_loss(*critic_loss_args)

        # backward and grad clip
        total_critic_loss = critic_loss + constraint_critic_loss
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm)
        torch.nn.utils.clip_grad_norm_(self.constraint_critic.parameters(), max_norm=self.grad_norm)

        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()
        update_ema(self.critic_target, self.critic, self.tau)
        update_ema(self.constraint_critic_target, self.constraint_critic, self.tau)
        return critic_loss.detach(), constraint_critic_loss.detach()

    def actor_loss(self, obs, include_alpha=True):
        distr = self.actor.distr(obs)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        actor_Q = self.critic.random_choice(obs, action)
        alpha = self.alpha
        uncstr_actor_loss = torch.mean(alpha.detach() * log_prob - actor_Q)

        # ----- constrained part ----- #
        if self.constrained_fcn == 'reachability':
            assert self.constraint_critic(obs, action).size(1) == self.con_dim
            actor_Qc, _ = torch.max(self.constraint_critic(obs, action), dim=1)
            # actor_Qc = actor_Qc + (actor_Qc>0).float() * self.penalty_offset
        else:
            assert self.constraint_critic(obs, action).size(1) == 1
            actor_Qc = self.constraint_critic(obs, action)
        if self.mlp_multiplier:
            with torch.no_grad():
                action_safe = self.actor_safe.act(obs, eval=True)
                safe_Qc, _ = torch.max(self.constraint_critic(obs, action_safe), dim=1)
                # lams = torch.max(self.multiplier(obs, action), (actor_Qc>0)*19.0).detach()
                lams = self.multiplier(obs, safe_Qc)
            assert lams.shape == actor_Qc.shape
        else:
            # lams = self.lam.detach()
            lams = self.fixed_multiplier
            actor_Qc = torch.clamp(actor_Qc, min=self.penalty_lb, max=self.penalty_ub)
        cstr_actor_loss = torch.mean(torch.mul(lams, actor_Qc))
        # ----- constrained part end ----- #

        # ----- safe actor loss ----- #
        if self.constrained_fcn == 'reachability':
            distr_safe = self.actor_safe.distr(obs)
            action_safe = distr_safe.rsample()
            assert self.constraint_critic(obs, action_safe).size(1) == self.con_dim
            actor_safe_Qc, _ = torch.max(self.constraint_critic(obs, action_safe), dim=1)
            actor_safe_loss = torch.mean(actor_safe_Qc)
        # ----- safe actor loss end ----- #

        actor_loss = uncstr_actor_loss + cstr_actor_loss
        losses = [actor_loss]
        if include_alpha:
            alpha_coefficient = self.log_alpha if self.use_log_alpha_loss else alpha
            alpha_loss = -alpha_coefficient * torch.mean(log_prob.detach() + self.target_entropy)
            losses += [alpha_loss]
        if self.constrained_fcn == 'reachability':
            losses += [actor_safe_loss]
        
        return losses

    def update_actor_and_alpha(self, obs):
        losses = self.actor_loss(obs, include_alpha=self.autotune_alpha)
        optimizers = [self.actor_optimizer, self.alpha_optimizer, self.actor_safe_optimizer] if self.autotune_alpha else \
                     [self.actor_optimizer, self.actor_safe_optimizer]
        assert len(losses) == len(optimizers)
        for i, (loss, optimizer) in enumerate(zip(losses, optimizers)):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if i == 0:  # for actor only: clip grad
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
            if i == 2:
                torch.nn.utils.clip_grad_norm_(self.actor_safe.parameters(), max_norm=self.grad_norm)
            optimizer.step()
            if i == 0:  # for actor only: lr schedule
                self.actor_lr_scheduler.step()
            if i == 2:
                self.actor_safe_lr_scheduler.step()
    
    def multiplier_loss(self, obs):
        distr = self.actor.distr(obs)
        action = distr.rsample()
        
        if self.constrained_fcn == 'reachability':
            assert self.constraint_critic(obs, action).size(1) == self.con_dim
            actor_Qc, _ = torch.max(self.constraint_critic(obs, action), dim=1)
        else:
            assert self.constraint_critic(obs, action).size(1) == 1
            actor_Qc = self.constraint_critic(obs, action)

        penalty = torch.clamp(
            actor_Qc - self.constraint_threshold,
            min=self.penalty_lb, max=self.penalty_ub
        )
        # penalty = penalty + (penalty>0).float() * self.penalty_offset
        if self.mlp_multiplier:
            action_safe = self.actor_safe.act(obs, eval=True)
            with torch.no_grad():
                safe_Qc, _ = torch.max(self.constraint_critic(obs, action_safe), dim=1)
            lams = self.multiplier(obs, safe_Qc)
            assert lams.shape == penalty.shape
            lams_safe = torch.mul(safe_Qc<=0, lams)
            lams_unsafe = torch.mul(safe_Qc>0, lams)
            
            '''Special lam loss
            For safe states, learn their lam: 0 or finite vlaues
            For unsafe states, want their lams to be close to max_multiplier - 1, a large penalty
                why - 1: because the upperbound of lams is max_multiplier, to avoid grad vanishing
            '''
            lam_loss = -0.5 * torch.mean(torch.mul(lams_safe, penalty.detach())) + \
                       self.criterion(lams_unsafe, (safe_Qc>0) * (self.max_multiplier - 1))
        else:
            lams = self.lam
            lam_loss = -torch.mean(torch.mul(lams, penalty.detach()))

        return lam_loss

    def update_multiplier(self, obs):
        losses = self.multiplier_loss(obs)
        self.multiplier_optimizer.zero_grad()
        losses.backward()
        if self.mlp_multiplier:
            torch.nn.utils.clip_grad_norm_(self.multiplier.parameters(), max_norm=self.grad_norm)
        self.multiplier_optimizer.step()
        if self.mlp_multiplier:
            self.multiplier_lr_scheduler.step()

    def update(self, replay_buffer):
        assert self.critic_update_multiplier >= 1
        for _ in range(self.critic_update_multiplier):
            samples = replay_buffer.sample(self.batch_size)
            self.update_critic(*samples)
        self.update_actor_and_alpha(samples[0])
        self.total_updates += 1