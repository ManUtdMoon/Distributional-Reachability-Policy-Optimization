from gym import register
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from .safe_env_spec import SafeEnv, interval_barrier
from gym.utils.ezpickle import EzPickle
import numpy as np


class SafeInvertedPendulumEnv(InvertedPendulumEnv):

    def __init__(self, threshold=0.2, task='upright'):
        self.threshold = threshold
        self.task = task
        self.margin = 0.1
        self._max_episode_steps = 1000
        super().__init__()
        EzPickle.__init__(self, threshold=threshold, task=task)  # deepcopy calls `get_state`

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
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
            constraint_value = max(
                next_state[1] - self.threshold,
                -next_state[1] - self.threshold
            )
            violation = not (constraint_value <= 0.)

            soft_constraint_value = max(
                next_state[1] - self.threshold - self.margin,
                -next_state[1] - self.threshold - self.margin
            )
            done = not (soft_constraint_value <= 0.)
        elif self.task == 'swing':
            assert 0
            reward = next_state[1]**2
        elif self.task == 'move':
            assert 0
            reward = next_state[0]**2
        else:
            assert 0
        
        info = dict(
            violation=violation,
            constraint_value=constraint_value
        )
        return next_state, reward, done, info

    def check_done(self, states: np.array):
        larger_violations = np.logical_not(self._soft_constraint_values(states) <= 0.)
        assert len(larger_violations.shape) == 1
        return larger_violations
    
    def check_violation(self, states: np.array):
        violations = np.logical_not(self._constraint_values(states) <= 0.)
        assert len(violations.shape) == 1
        return violations

    def _soft_constraint_values(self, states: np.array):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2
        if self.task == 'upright':
            ths = states[:, 1]
            return np.maximum(
                ths - self.threshold - self.margin,
                - ths - self.threshold - self.margin
            )
        elif self.task == 'move':
            assert 0
    
    def _constraint_values(self, states: np.array):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2
        if self.task == 'upright':
            ths = states[:, 1]
            return np.maximum(
                ths - self.threshold,
                - ths - self.threshold
            )
        elif self.task == 'move':
            assert 0