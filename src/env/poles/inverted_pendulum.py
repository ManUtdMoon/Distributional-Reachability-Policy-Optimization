from gym import register
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from .safe_env_spec import SafeEnv, interval_barrier
from gym.utils.ezpickle import EzPickle
import numpy as np


class SafeInvertedPendulumEnv(InvertedPendulumEnv, SafeEnv):
    episode_unsafe = False

    def __init__(self, threshold=0.2, task='upright'):
        self.threshold = threshold
        self.task = task
        super().__init__()
        EzPickle.__init__(self, threshold=threshold, task=task)  # deepcopy calls `get_state`

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        self.episode_unsafe = False
        return self._get_obs()

    def _get_obs(self):
        return super()._get_obs().astype(np.float32)

    def step(self, a):
        a = np.clip(a, -1, 1)
        next_state, _, done, info = super().step(a)
        # reward = (next_state[0]**2 + next_state[1]**2)  # + a[0]**2 * 0.01
        # reward = next_state[1]**2  # + a[0]**2 * 0.01

        if self.task == 'upright':
            reward = -next_state[1]**2
        elif self.task == 'swing':
            reward = next_state[1]**2
        elif self.task == 'move':
            reward = next_state[0]**2
        else:
            assert 0
        
        if abs(next_state[..., 1]) > self.threshold:
            self.episode_unsafe = True
            reward -= 3
        info['episode.unsafe'] = self.episode_unsafe
        return next_state, reward, False, info

    def is_state_safe(self, states):
        return states[..., 1].abs() <= self.threshold

    def barrier_fn(self, states):
        return interval_barrier(states[..., 1], -self.threshold, self.threshold)

    def reward_fn(self, states, actions, next_states):
        return -(next_states[..., 0]**2 + next_states[..., 1]**2) - actions[..., 0]**2 * 0.01
