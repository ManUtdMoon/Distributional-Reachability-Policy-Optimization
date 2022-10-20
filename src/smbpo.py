import numpy as np
import torch
from tqdm import trange

from .config import BaseConfig, Configurable
from .dynamics import BatchedGaussianEnsemble
from .env.batch import ProductEnv
from .env.util import env_dims, get_max_episode_steps
from .log import default_log as log, TabularLog
from .policy import UniformPolicy
from .sampling import sample_episodes_batched, SafetySampleBuffer, ConstraintSafetySampleBuffer
from .ssac import SSAC
from .torch_util import Module, DummyModuleWrapper, device, torchify, random_choice, gpu_mem_info, deciles
from .util import pythonic_mean, batch_map


N_EVAL_TRAJ = 10
LOSS_AVERAGE_WINDOW = 10


class SMBPO(Configurable, Module):
    class Config(BaseConfig):
        sac_cfg = SSAC.Config()
        model_cfg = BatchedGaussianEnsemble.Config()
        model_initial_steps = 10000
        model_steps = 2000
        model_update_period = 250   # how many steps between updating the models
        save_trajectories = False
        horizon = 10
        alive_bonus = 1.0   # alternative: positive, rather than negative, reinforcement
        buffer_min = 5000
        buffer_max = 10**6
        steps_per_epoch = 1000
        rollout_batch_size = 100
        solver_updates_per_step = 10
        real_fraction = 0.1
        action_clip_gap = 1e-6  # for clipping to promote numerical instability in logprob
        reward_scale = 1.
        mode = 'train'
        constraint_scale = 10.
        constraint_offset = 0.
        safe_shield = True
        safe_shield_threshold = -0.1
        eval_shield_threshold = -0.05
        eval_shield_type = "linear"

    def __init__(self, config, env_factory, data, epochs):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.data = data
        self.episode_log = TabularLog(log.dir, 'episodes.csv')

        self.real_env = env_factory()
        if self.mode == 'train':
            self.eval_env = ProductEnv([env_factory(id=i) for i in range(N_EVAL_TRAJ)])
        elif self.mode == 'test':
            self.eval_env = ProductEnv([env_factory(id=i) for i in range(1)])
        else:
            log.message(f'Invalid SMBPO mode')

        self.state_dim, self.action_dim, self.con_dim = env_dims(self.real_env)

        self.check_done = lambda states: torchify(self.real_env.check_done(states.cpu().numpy()))
        self.check_violation = lambda states: torchify(self.real_env.check_violation(states.cpu().numpy()))
        self.get_constraint_value = lambda states: torchify(self.real_env.get_constraint_values(states.cpu().numpy()))
        self.get_reward = lambda states, actions: torchify(self.real_env.get_rewards(states.cpu().numpy(), actions.cpu().numpy()))

        self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, self.state_dim, self.action_dim)
        self.solver = SSAC(self.sac_cfg, self.state_dim, self.action_dim, self.con_dim, 
                           self.horizon, epochs, self.steps_per_epoch, self.solver_updates_per_step,
                           self.constraint_scale, env_factory,
                           self.model_ensemble)

        self.replay_buffer = self._create_buffer(self.buffer_max)
        self.virt_buffer = self._create_buffer(self.buffer_max)

        self.uniform_policy = UniformPolicy(self.real_env)

        self.register_buffer('episodes_sampled', torch.tensor(0))
        self.register_buffer('steps_sampled', torch.tensor(0))
        self.register_buffer('n_violations', torch.tensor(0))
        self.register_buffer('epochs_completed', torch.tensor(0))

        self.recent_critic_losses = []
        self.recent_cons_critic_losses = []

        self.stepper = None

    @property
    def actor(self):
        return self.solver.actor

    @property
    def constraint_critic(self):
        return self.solver.constraint_critic

    @property
    def actor_safe(self):
        return self.solver.actor_safe

    def _create_buffer(self, capacity):
        kwargs = {'con_dim': self.con_dim}
        buffer = ConstraintSafetySampleBuffer(self.state_dim, self.action_dim, capacity, **kwargs)
        buffer.to(device)
        return DummyModuleWrapper(buffer)

    def _log_tabular(self, row):
        for k, v in row.items():
            self.data.append(k, v, verbose=True)
        self.episode_log.row(row)

    def step_generator(self):
        max_episode_steps = get_max_episode_steps(self.real_env)
        episode = self._create_buffer(max_episode_steps)
        state = self.real_env.reset()
        while True:
            t = self.steps_sampled.item()
            if t >= self.buffer_min:
                policy = self.actor
                policy_safe = self.actor_safe
                constraint_critic = self.constraint_critic
                if t % self.model_update_period == 0:
                    self.update_models(self.model_steps)
                self.rollout_and_update()
                action = policy.act1(state, eval=False)

                # -------- safety shield ----------- #
                if self.safe_shield:
                    qc = self.solver._get_qc(
                        constraint_critic(
                            state.unsqueeze(0),
                            action.unsqueeze(0),
                            uncertainty=self.solver.distributional_qc
                        )
                    )
                    if qc > self.safe_shield_threshold:
                        action = policy_safe.act1(state, eval=True)
                    
                    # search for safe action if actor_safe is not safe, but maybe useless
                    # qc_safe = torch.max(constraint_critic(state.unsqueeze(0), action.unsqueeze(0)))
                    # if qc_safe > self.safe_shield_threshold:
                    #     for _ in range(100):
                    #         action_random = policy_safe.act1(state, eval=False)
                    #         qc_random = torch.max(constraint_critic(state.unsqueeze(0), action_random.unsqueeze(0)))
                    #         if qc_random <= self.safe_shield_threshold:
                    #             print("find safe action!!!!!!!!")
                    #             action = action_random
                    #             break
                    #         if _ == 99:
                    #             print("failed to find safe action!!!!!!!!")
                # -------- safety shield end -------- #

            else:
                policy = self.uniform_policy
                action = policy.act1(state, eval=False)
            next_state, reward, done, info = self.real_env.step(action)
            violation = info['violation']
            constraint_value = torch.tensor(info['constraint_value'], dtype=torch.float)
            assert done == self.check_done(next_state.unsqueeze(0)).item(), \
                print(done, self.check_done(next_state.unsqueeze(0)).item(), next_state)
            assert violation == self.check_violation(next_state.unsqueeze(0)).item(), \
                print(violation, self.check_violation(next_state.unsqueeze(0)).item(), next_state)
            # print('constraint_value', constraint_value)
            # print('get_constraint_value', self.get_constraint_value(next_state.unsqueeze(0)).cpu())
            # print(constraint_value.numpy() == self.get_constraint_value(next_state.unsqueeze(0)).cpu().numpy())
            assert torch.all(torch.isclose(constraint_value, self.get_constraint_value(next_state.unsqueeze(0)).cpu(), atol=1e-05)), \
                print(constraint_value.numpy() - self.get_constraint_value(next_state.unsqueeze(0)).cpu().numpy())
            for buffer in [episode, self.replay_buffer]:
                buffer.append(states=state, actions=action, next_states=next_state,
                              rewards=reward, dones=done, violations=violation,
                              constraint_values=constraint_value)
            self.steps_sampled += 1

            if done or (len(episode) == max_episode_steps):
                episode_return = episode.get('rewards').sum().item()
                episode_length = len(episode)
                episode_return_plus_bonus = episode_return + episode_length * self.alive_bonus
                episode_safe = not episode.get('violations').any()
                self.episodes_sampled += 1
                if not episode_safe:
                    self.n_violations += episode.get('violations').sum()

                self._log_tabular({
                    'episodes sampled': self.episodes_sampled.item(),
                    'total violations': self.n_violations.item(),
                    'steps sampled': self.steps_sampled.item(),
                    'collect return': episode_return,
                    'collect return (+bonus)': episode_return_plus_bonus,
                    'collect length': episode_length,
                    'collect safe': episode_safe,
                    # **self.evaluate()
                })

                if self.save_trajectories:
                    episode_num = self.episodes_sampled.item()
                    save_path = self.episodes_dir/f'episode-{episode_num}.h5py'
                    episode.save_h5py(save_path)
                    log.message(f'Saved episode to {save_path}')

                episode = self._create_buffer(max_episode_steps)
                state = self.real_env.reset()
            else:
                if self.steps_sampled % max_episode_steps == 0:
                    self._log_tabular({
                        'episodes sampled': self.episodes_sampled.item(),
                        'total violations': self.n_violations.item(),
                        'steps sampled': self.steps_sampled.item(),
                        'collect return': None,
                        'collect return (+bonus)': None,
                        'collect length': None,
                        'collect safe': None,
                        # **self.evaluate()
                    })
                state = next_state

            yield t

    def update_models(self, model_steps):
        log.message(f'Fitting models @ t = {self.steps_sampled.item()}')
        model_losses = self.model_ensemble.fit(self.replay_buffer, steps=model_steps)
        start_loss_average = np.mean(model_losses[:LOSS_AVERAGE_WINDOW])
        end_loss_average = np.mean(model_losses[-LOSS_AVERAGE_WINDOW:])
        log.message(f'Loss statistics:')
        log.message(f'\tFirst {LOSS_AVERAGE_WINDOW}: {start_loss_average}')
        log.message(f'\tLast {LOSS_AVERAGE_WINDOW}: {end_loss_average}')
        log.message(f'\tDeciles: {deciles(model_losses)}')

        buffer_rewards = self.replay_buffer.get('rewards')
        r_min = buffer_rewards.min().item() + self.alive_bonus
        r_max = buffer_rewards.max().item() + self.alive_bonus
        self.solver.update_r_bounds(r_min, r_max)

    def rollout(self, policy, initial_states=None):
        if initial_states is None:
            initial_states = random_choice(self.replay_buffer.get('states'), size=self.rollout_batch_size)
        buffer = self._create_buffer(self.rollout_batch_size * self.horizon)
        states = initial_states
        for t in range(self.horizon):
            with torch.no_grad():
                actions = policy.act(states, eval=False)
                next_states, rewards = self.model_ensemble.sample(states, actions)
                rewards = self.get_reward(next_states, actions)
            dones = self.check_done(next_states)
            violations = self.check_violation(next_states)
            constraint_values = self.get_constraint_value(next_states)
            buffer.extend(states=states, actions=actions, next_states=next_states,
                          rewards=rewards, dones=dones, violations=violations, constraint_values=constraint_values)
            continues = ~(dones)
            if continues.sum() == 0:
                break
            states = next_states[continues]

        self.virt_buffer.extend(**buffer.get(as_dict=True))
        return buffer

    def update_solver(self, update_actor=True, update_multiplier=False):
        solver = self.solver
        n_real = int(self.real_fraction * solver.batch_size)
        real_samples = self.replay_buffer.sample(n_real)
        virt_samples = self.virt_buffer.sample(solver.batch_size - n_real)
        combined_samples = [
            torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)
        ]

        # reward preprocess
        REWARD_INDEX = 3
        assert combined_samples[REWARD_INDEX].ndim == 1
        if self.reward_scale != 0:
            combined_samples[REWARD_INDEX] = combined_samples[REWARD_INDEX] * self.reward_scale
        if self.alive_bonus != 0:    
            combined_samples[REWARD_INDEX] = combined_samples[REWARD_INDEX] + self.alive_bonus
        CONSTRAINT_VALUE_INDEX = 6
        combined_samples[CONSTRAINT_VALUE_INDEX] = combined_samples[CONSTRAINT_VALUE_INDEX] * self.constraint_scale
        combined_samples[CONSTRAINT_VALUE_INDEX] = combined_samples[CONSTRAINT_VALUE_INDEX] + \
            (combined_samples[CONSTRAINT_VALUE_INDEX]>0).float() * self.constraint_offset

        # learning
        critic_loss, constraint_critic_loss = solver.update_critic(*combined_samples)
        self.recent_critic_losses.append(critic_loss)
        self.recent_cons_critic_losses.append(constraint_critic_loss)
        if update_actor:
            solver.update_actor_and_alpha(combined_samples[0])
        if update_multiplier:
            solver.update_multiplier(combined_samples[0])

    def rollout_and_update(self):
        self.rollout(self.actor)
        for step in range(self.solver_updates_per_step):
            update_multiplier = True if step % self.sac_cfg.multiplier_update_interval == 0 else \
                                False
            update_actor = True if step % self.sac_cfg.actor_update_interval == 0 else \
                           False
            self.update_solver(
                update_actor=update_actor,
                update_multiplier=update_multiplier
            )

    def setup(self):
        if self.save_trajectories:
            self.episodes_dir = log.dir/'episodes'
            self.episodes_dir.mkdir(exist_ok=True)

        episodes_to_load = self.episodes_sampled.item()
        if episodes_to_load > 0:
            log.message(f'Loading existing {episodes_to_load} episodes')
            for i in trange(1, self.episodes_sampled + 1):
                episode = ConstraintSafetySampleBuffer.from_h5py(self.episodes_dir/f'episode-{i}.h5py')
                self.replay_buffer.extend(*episode.get())

        assert len(self.replay_buffer) == self.steps_sampled

        self.stepper = self.step_generator()

        if len(self.replay_buffer) < self.buffer_min:
            log.message(f'Collecting initial data')
            while len(self.replay_buffer) < self.buffer_min:
                next(self.stepper)
            log.message('Initial model training')
            self.update_models(self.model_initial_steps)
        # learn qc*, actor_safe
        log.message(f'Collecting initial virtual data')
        while len(self.virt_buffer) < self.buffer_min:
            self.rollout(self.uniform_policy)
        log.message(f'Setup done!')

    def epoch(self):
        for _ in trange(self.steps_per_epoch):
            next(self.stepper)
        self.log_statistics()
        self.epochs_completed += 1

    def evaluate_models(self):
        states, actions, next_states = self.replay_buffer.get('states', 'actions', 'next_states')
        state_std = states.std(dim=0)
        state_std[state_std<1e-7] = 1.0
        with torch.no_grad():
            predicted_states = batch_map(lambda s, a: self.model_ensemble.means(s, a)[0],
                                         [states, actions], cat_dim=1)
        for i in range(self.model_cfg.ensemble_size):
            errors = torch.norm((predicted_states[i] - next_states) / (state_std + 1e-7), dim=1)
            log.message(f'Model {i+1} error deciles: {deciles(errors)}')

    def log_statistics(self):
        self.evaluate_models()

        avg_critic_loss = pythonic_mean(self.recent_critic_losses)
        log.message(f'Average recent critic loss: {avg_critic_loss}')
        self.data.append('critic loss', avg_critic_loss)
        self.recent_critic_losses.clear()

        avg_cons_critic_loss = pythonic_mean(self.recent_cons_critic_losses)
        log.message(f'Average recent constraint critic loss: {avg_cons_critic_loss}')
        self.data.append('constraint critic loss', avg_cons_critic_loss)
        self.recent_cons_critic_losses.clear()

        log.message('Buffer sizes:')
        log.message(f'\tReal: {len(self.replay_buffer)}')
        log.message(f'\tVirtual: {len(self.virt_buffer)}')

        real_states, real_actions, real_violations = self.replay_buffer.get('states', 'actions', 'violations')
        virt_states, virt_violations = self.virt_buffer.get('states', 'violations')
        virt_actions = self.actor.act(virt_states, eval=True).detach()
        sa_data = {
            'real (done)': (real_states[real_violations], real_actions[real_violations]),
            'real (~done)': (real_states[~real_violations], real_actions[~real_violations]),
            'virtual (done)': (virt_states[virt_violations], virt_actions[virt_violations]),
            'virtual (~done)': (virt_states[~virt_violations], virt_actions[~virt_violations])
        }
        for which, (states, actions) in sa_data.items():
            if len(states) == 0:
                mean_q = None
                mean_qc = None
            else:
                with torch.no_grad():
                    qs = batch_map(lambda s, a: self.solver.critic.mean(s, a), [states, actions])
                    qcs = batch_map(
                        lambda s, a: self.solver.constraint_critic(s, a),
                        [states, actions]
                    )
                    if self.sac_cfg.distributional_qc:
                        qcs_std = batch_map(
                            lambda s, a: self.solver.constraint_critic(s, a, sample=True)[1],
                            [states, actions]
                        )
                    if self.sac_cfg.constrained_fcn == 'reachability':
                        a_safe = batch_map(
                            lambda s: self.solver.actor_safe.act(s, eval=True).detach(),
                            [states]
                        )
                        safe_qcs = batch_map(
                            lambda s, a: self.solver._get_qc(self.solver.constraint_critic(s, a)),
                            [states, a_safe]
                        )
                        if self.sac_cfg.mlp_multiplier:
                            lams = batch_map(
                                lambda s, qc: self.solver.multiplier(s, qc),
                                [states, safe_qcs]
                            )
                            mean_lam = lams.mean()

                    mean_q = qs.mean()
                    if self.sac_cfg.constrained_fcn == 'reachability':
                        qcs = self.solver._get_qc(qcs)
                    mean_qc = qcs.mean()
                    if self.sac_cfg.distributional_qc:
                        mean_qc_std = qcs_std.mean()
            log.message(f'Average Q {which}: {mean_q}')
            self.data.append(f'Average Q {which}', mean_q)
            log.message(f'Average Qc {which}: {mean_qc}')
            self.data.append(f'Average Qc {which}', mean_qc)
            if self.sac_cfg.distributional_qc:
                log.message(f'Average Qc std {which}: {mean_qc_std}')
                self.data.append(f'Average Qc std {which}', mean_qc_std)
            if self.sac_cfg.mlp_multiplier:
                log.message(f'Average Lambda {which}: {mean_lam}')
                self.data.append(f'Average Lambda {which}', mean_lam)
        
        if not self.sac_cfg.mlp_multiplier:
            mean_lam = self.solver.lam.detach().item()
            log.message(f'Average Lambda: {mean_lam}')
            self.data.append(f'Average Lambda', mean_lam)

        if torch.cuda.is_available():
            log.message(f'GPU memory info: {gpu_mem_info()}')

    def evaluate(self):
        eval_traj = sample_episodes_batched(self.eval_env, self.solver, N_EVAL_TRAJ, 
                                            eval=True, safe_shield_threshold=self.eval_shield_threshold, shield_type=self.eval_shield_type)

        lengths = [len(traj) for traj in eval_traj]
        length_mean, length_std = float(np.mean(lengths)), float(np.std(lengths))

        returns = [traj.get('rewards').sum().item() for traj in eval_traj]
        return_mean, return_std = float(np.mean(returns)), float(np.std(returns))

        violations = [traj.get('violations').sum().item() for traj in eval_traj]
        violation_mean = float(np.mean(violations))

        return {
            'eval return mean': return_mean,
            'eval return std': return_std,
            'eval length mean': length_mean,
            'eval length std': length_std,
            'eval violation mean': violation_mean
        }