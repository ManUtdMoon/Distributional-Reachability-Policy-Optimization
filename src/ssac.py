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


class ConstraintCritic(Configurable, Module):
    class Config(BaseConfig):
        trunk_layers = 2
        head_layers = 1
        hidden_dim = 256
        log_std_min = -2.
        log_std_max = 4.
        std_ratio = 1. # 1 2 3, shift how many stds

    def __init__(self, config, state_dim, action_dim, output_dim, output_activation=None):
        Configurable.__init__(self, config)
        Module.__init__(self)
        trunk_dims = [state_dim + action_dim] + [self.hidden_dim] * self.trunk_layers
        head_dims = [self.hidden_dim] * (self.head_layers + 1) + [output_dim]
        self.trunk = mlp(trunk_dims, output_activation="relu")
        self.mean_head = mlp(head_dims, squeeze_output=True)
        self.log_std_head = mlp(head_dims, squeeze_output=True)
    
    def forward(self, state, action, uncertainty=False, sample=False):
        sa = torch.cat([state, action], -1)
        shared_hidden = self.trunk(sa)
        mean = self.mean_head(shared_hidden)

        if (not uncertainty) and (not sample):
            return mean
        assert not (uncertainty and sample), \
            print("Uncertainty bound and sample cannot be True simulnateously.")

        log_std = self.log_std_head(shared_hidden)
        log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        log_std = self.log_std_min + F.softplus(log_std - self.log_std_min)
        std = log_std.exp()

        # perform reparameterization
        noise = torch.randn_like(std)
        if uncertainty:
            # noise = noise.abs()
            # if self.std_ratio:
            #     noise = torch.clamp(noise, max=self.std_ratio).to(device)
            qc_sample = mean + torch.mul(self.std_ratio, std)
            return qc_sample
        elif sample:
            noise = torch.clamp(noise, -2., 2.).to(device)
            qc_sample = mean + torch.mul(noise, std)
            return mean, std, qc_sample
        else:
            raise NotImplementedError("Unknown Qc forward type.")


class MLPMultiplier(Configurable, Module):
    class Config(BaseConfig) :
        hidden_layers = 2
        hidden_dim = 256
        upper_bound = 50.

    def __init__(self, config, state_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + 1, *([self.hidden_dim] * self.hidden_layers), 1]
        self.lam = mlp(dims, activation='tanh', output_activation='identity', squeeze_output=True)
    
    def forward(self, state, Qc):
        states_aug = torch.cat([state, Qc.unsqueeze(-1)], -1)
        lam = self.upper_bound/2. * \
            (1. + torch.tanh( self.lam(states_aug)/self.upper_bound*2 ))
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
        actor_lr = 1e-4
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
        constrained_fcn = 'cost'
        mlp_multiplier = False
        penalty_lb = -1.0
        penalty_ub = 100.
        # penalty_offset = 1.0
        fixed_multiplier = 15.0
        multiplier_update_interval = 5

        lam_epsilon = 1.0
        qc_under_uncertainty = False
        qc_td_bound = 5.
        distributional_qc = False

        # conservative safety critic
        enable_csc = False
        csc_weight_coefficient = 0.5

    def __init__(self, config, state_dim, action_dim, con_dim, 
                 horizon, epochs, steps_per_epoch, solver_updates_per_step,
                 constraint_scale, env_factory, model_ensemble,
                 optimizer_factory=OPTIMIZER):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.con_dim = con_dim
        self.horizon = horizon
        self.violation_cost = 0.0
        self.updates_per_training = epochs * steps_per_epoch * solver_updates_per_step
        self.lam_updates_num = int(self.updates_per_training / self.multiplier_update_interval)
        self.actor_updates_num = int(self.updates_per_training / self.actor_update_interval)

        # -------- get env, just to use the get_constraint_value -------- #
        self.env = env_factory()
        self.check_done = lambda states: torchify(self.env.check_done(states.cpu().numpy()))
        self.check_violation = lambda states: torchify(self.env.check_violation(states.cpu().numpy()))
        self.get_constraint_value = lambda states: \
            torchify(self.env.get_constraint_values(states.cpu().numpy()) * constraint_scale)

        self.model_ensemble = model_ensemble

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
            self.multiplier = MLPMultiplier(self.mlp_multiplier_cfg, state_dim)
            self.multiplier_optimizer = optimizer_factory(self.multiplier.parameters(), lr=self.multiplier_lr, weight_decay=1e-4)
            self.multiplier_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.multiplier_optimizer,
                T_max=self.lam_updates_num,
                eta_min=self.multiplier_lr_end
            )
        else:
            self.multiplier = nn.parameter.Parameter(
                torch.tensor(0., device=device, dtype=torch.float)  # todo: a larger initial multiplier
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
    
    def compute_cons_target(self, obs, action, next_obs, done, violation, constraint_value):
        with torch.no_grad():
            if self.constrained_fcn == 'cost':
                distr = self.actor.distr(next_obs)
                next_action = distr.sample()
                next_qc_value = self.constraint_critic_target(next_obs, next_action)
                qc = violation.float() + self.discount * (1. - done.float()) * next_qc_value
            elif self.constrained_fcn == 'reachability':
                """get targetQc with s, a, s' under uncertainty
                w/o uncertainty:
                    qc_target = done * h(s)
                                (1 - done) * { (1-\gamma) * h(s) 
                                              + \gamma * max[h(s), qc(s')] ) }
                w/ uncertainty:
                    We won't use the s' in the buffer because it is deterministic.
                    However, we need to model the uncertainty in current model ensembles.
                    Hence, we need rollout a samples distribution with ensembles and we 
                    denote them as {s'[i]}_{i=1~model_ensemble.ensemble_size}
                    a = \pi(s)
                    s'[i] = f(s, a)
                    qc'[i] = h(s') if s' done else qc(s'[i])
                    qc' = max_i qc'[i]

                    We need to consider the following cases:
                               s       s'
                    case 1   done      /
                    case 2   ~done    done
                    case 3   ~done    ~done
                    
                    case 1:
                        qc_target = h(s)
                    case 2 & 3:
                        qc_target = max{h(s), qc'}
                """
                if self.qc_under_uncertainty:
                    if self.distributional_qc:
                        distr = self.actor_safe.distr(next_obs)
                        next_action = distr.sample()
                        _, _, next_qc_sample = self.constraint_critic_target(
                            next_obs, next_action, sample=True
                        )
                        qc_mean = self.constraint_critic(obs, action)
                        qc_nonterminal = (1. - self.discount) * constraint_value + \
                                         self.discount * torch.maximum(constraint_value, next_qc_sample)
                        dones = done.tile((self.con_dim, 1)).t().squeeze().float()
                        target_qc_unbounded = qc_nonterminal * (1 - dones) + constraint_value * dones
                        difference = torch.clamp(
                            target_qc_unbounded - qc_mean,
                            min=-self.qc_td_bound, max=self.qc_td_bound
                        )
                        target_qc_bounded = difference + qc_mean

                        value_shape = (self.batch_size,) if self.con_dim == 1 \
                                 else (self.batch_size, self.con_dim)
                        assert next_qc_sample.shape == \
                            qc_mean.shape == \
                            target_qc_bounded.shape ==\
                            value_shape
                        return target_qc_unbounded, target_qc_bounded
                    else:
                        # we denote elite_batch_sth. as en_ba_sth. for short
                        # ----- start: robust Qc: choose the largest Qc ----- #
                        # en_ba_next_obs, _ = self.model_ensemble.elite_samples(obs, action)
                        # en_ba_done = self.check_done(en_ba_next_obs)  # (E, B)
                        # en_ba_distr = self.actor_safe.distr(en_ba_next_obs)
                        # en_ba_next_act = en_ba_distr.sample()  # (E, B, d_a)
                        # en_ba_next_qc = self.constraint_critic_target(
                        #     en_ba_next_obs, en_ba_next_act
                        # )  # (E, B, con_dim)
                        
                        # en_ba_con_value = constraint_value.repeat(self.model_ensemble.num_elites, 1, 1)
                        # en_ba_qc_nonterminal = (1 - self.discount) * constraint_value +\
                        #                             self.discount * torch.maximum(constraint_value, en_ba_next_qc)

                        # en_ba_dones = en_ba_done.tile((self.con_dim, 1, 1)).permute(1, 2, 0)  # (E, B, con_dim)
                        # assert en_ba_qc_nonterminal.shape == en_ba_con_value.shape == en_ba_dones.shape ==\
                        #     (self.model_ensemble.num_elites, self.batch_size, self.con_dim)

                        # en_ba_qc = torch.where(en_ba_dones, en_ba_con_value, en_ba_qc_nonterminal)  # (E, B, con_dim)
                        # qc = torch.topk(en_ba_qc, k=2, dim=0)[0][-1]  # (B, con_dim)
                        # assert qc.shape == (self.batch_size, self.con_dim)
                        # ----- end ----- #
                        
                        # ----- start: robust Qc: select a model randomly----- #
                        next_obs, _ = self.model_ensemble.sample(obs, action)
                        ba_done = self.check_done(next_obs)  # (B)
                        ba_distr = self.actor_safe.distr(next_obs)
                        ba_next_act = ba_distr.sample()  # (B, d_a)
                        qc_next = self.constraint_critic_target(
                            next_obs, ba_next_act
                        )  # (B, con_dim)
                        dones = ba_done.tile((self.con_dim, 1)).t().squeeze()
                        qc_nonterminal = (1. - self.discount) * constraint_value +\
                                            self.discount * torch.maximum(constraint_value, qc_next)
                        qc = torch.where(dones, constraint_value, qc_nonterminal)
                        assert qc_next.shape == dones.shape == constraint_value.shape, \
                            print(f"qc: {qc_next.shape}, dones: {dones.shape}, cons: {constraint_value.shape}")
                        # ----- end ----- #
                else:
                    distr = self.actor_safe.distr(next_obs)
                    next_action = distr.sample()
                    next_qc_value = self.constraint_critic_target(next_obs, next_action)
                    qc_nonterminal = (1. - self.discount) * constraint_value + \
                                           self.discount * torch.maximum(constraint_value, next_qc_value)
                    dones = done.tile((self.con_dim, 1)).t().squeeze().float()
                    qc = qc_nonterminal * (1 - dones.float()) + constraint_value * dones.float()
                    assert qc.shape == qc_nonterminal.shape
            else:
                raise NotImplementedError
            return qc

    def cons_critic_loss_given_target(self, obs, action, target, target_bounded=None):
        qcs, qcs_std, _ = self.constraint_critic(obs, action, sample=True)
        if self.distributional_qc:
            assert target_bounded is not None
            loss = torch.mean(
                torch.pow(qcs - target, 2) / (2 * torch.pow(qcs_std.detach(), 2)) +\
                torch.pow(qcs.detach() - target_bounded, 2) / (2 * torch.pow(qcs_std, 2)) +\
                torch.log(qcs_std)
            )
            return loss
        else:
            assert target_bounded is None
            if self.enable_csc:
                mse_loss = self.criterion(qcs, target)
                with torch.no_grad():
                    distr = self.actor.distr(obs)
                    action = distr.sample()
                qcs_pi = self.constraint_critic(obs, action)
                csc_loss = torch.mean(qcs - qcs_pi)
                return mse_loss + csc_loss * self.csc_weight_coefficient
            else:
                return self.criterion(qcs, target)

    def constraint_critic_loss(self, obs, action, next_obs, reward, done, violation, constraint_value):
        if self.distributional_qc:
            target, target_bounded = self.compute_cons_target(obs, action, next_obs, done, violation, constraint_value)
        else:
            target = self.compute_cons_target(obs, action, next_obs, done, violation, constraint_value)
            target_bounded = None
        return self.cons_critic_loss_given_target(obs, action, target, target_bounded)

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
            actor_Qc_ub_con_dim = self.constraint_critic(obs, action, uncertainty=self.distributional_qc)
            actor_Qc = self._get_qc(actor_Qc_ub_con_dim)
            # actor_Qc = actor_Qc + (actor_Qc>0).float() * self.penalty_offset
        else:
            actor_Qc = self.constraint_critic(obs, action)
        if self.mlp_multiplier:
            with torch.no_grad():
                action_safe = self.actor_safe.act(obs, eval=True)
                safe_Qc = self._get_qc(self.constraint_critic(obs, action_safe, uncertainty=self.distributional_qc))
                # lams = torch.max(self.multiplier(obs, action), (actor_Qc>0)*19.0).detach()
                lams = self.multiplier(obs, safe_Qc)
            assert lams.shape == actor_Qc.shape
        else:
            lams = self.lam.detach()
            # lams = self.fixed_multiplier
            actor_Qc = torch.clamp(actor_Qc, min=self.penalty_lb, max=self.penalty_ub)
        cstr_actor_loss = torch.mean(torch.mul(lams, actor_Qc))
        # ----- constrained part end ----- #

        # ----- safe actor loss ----- #
        if self.constrained_fcn == 'reachability':
            distr_safe = self.actor_safe.distr(obs)
            action_safe = distr_safe.rsample()
            actor_safe_Qc_con_dim = self.constraint_critic(obs, action_safe, uncertainty=self.distributional_qc)
            actor_safe_Qc = self._get_qc(actor_safe_Qc_con_dim)
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
        optimizers = [self.actor_optimizer]
        if self.autotune_alpha:
            optimizers.append(self.alpha_optimizer)
        if self.constrained_fcn == 'reachability':
            optimizers.append(self.actor_safe_optimizer)

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
            actor_Qc = self.constraint_critic(obs, action, uncertainty=self.distributional_qc)
            actor_Qc = self._get_qc(actor_Qc)
        else:
            actor_Qc = self.constraint_critic(obs, action)
            assert actor_Qc.size(0) == self.batch_size

        penalty = torch.clamp(
            actor_Qc - self.constraint_threshold,
            min=self.penalty_lb, max=self.penalty_ub
        )
        # penalty = penalty + (penalty>0).float() * self.penalty_offset
        if self.mlp_multiplier:
            action_safe = self.actor_safe.act(obs, eval=True)
            with torch.no_grad():
                safe_Qc = self._get_qc(self.constraint_critic(obs, action_safe, uncertainty=self.distributional_qc))
            lams = self.multiplier(obs, safe_Qc)
            assert lams.shape == penalty.shape
            lams_safe = torch.mul(safe_Qc<=0, lams)
            lams_unsafe = torch.mul(safe_Qc>0, lams)
            
            '''Special lam loss
            For safe states, learn their lam: 0 or finite vlaues
            For unsafe states, want their lams to be close to ub, a large penalty
                why (ub-\epsilon)): because the upperbound of lams is ub, to avoid grad vanishing
            '''
            lam_loss = -0.5 * torch.mean(torch.mul(lams_safe, penalty.detach())) + \
                       self.criterion(
                           lams_unsafe,
                           (safe_Qc>0) * (self.mlp_multiplier_cfg.upper_bound - self.lam_epsilon)
                       )
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
    
    def _get_qc(self, qc_con_dim):
        '''get qc with qc_con_dim 
            params:
                qc_con_dim: (batch, con_dim) when con_dim > 1 
                    else: (batch,)
            return:
                qc: (batch,)
        '''
        if self.con_dim > 1:
            assert qc_con_dim.size(-1) == self.con_dim
            return torch.max(qc_con_dim, dim=-1)[0]
        elif self.con_dim == 1:
            return qc_con_dim
