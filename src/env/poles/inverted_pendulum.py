import numpy as np

from gym import register
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.utils.ezpickle import EzPickle

from .constraints import BoundedConstraint, ConstrainedVariableType

class SafeInvertedPendulumEnv(InvertedPendulumEnv):

    def __init__(self, threshold=0.2, task='upright'):
        self.task = task
        self.th_threshold = threshold
        self.th_margin = 0.1
        
        self.x_threshold = 0.9
        self.x_margin = 0.1
        
        # for violation
        self.constraints = BoundedConstraint(
            4,  # we know dim_states in advance, and it is also an ugly approach
            lower_bounds=[-self.x_threshold, -self.th_threshold],
            upper_bounds=[self.x_threshold, self.th_threshold],
            constrained_variable=ConstrainedVariableType.STATE,
            active_dims=[0, 1]
        )

        # for done
        self.soft_constraints = BoundedConstraint(
            4,
            lower_bounds=[-(self.x_threshold + self.x_margin), -(self.th_threshold + self.th_margin)],
            upper_bounds=[self.x_threshold + self.x_margin, self.th_threshold + self.th_margin],
            constrained_variable=ConstrainedVariableType.STATE,
            active_dims=[0, 1]
        )

        self.con_dim = 4  # lb & ub

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
        elif self.task == 'swing':
            assert 0, 'UNDEFINED TASK'
            reward = next_state[1]**2
        elif self.task == 'move':
            reward = next_state[0]**2
        else:
            assert 0, 'UNDEFINED TASK'
        
        constraint_value = self._constraint_values(next_state)
        violation = np.any(constraint_value > 0.).item()

        info = dict(
            violation=violation,
            constraint_value=constraint_value
        )

        done = np.any(self._soft_constraint_values(next_state) > 0.).item()

        return next_state, reward, done, info

    def check_done(self, states: np.array):
        larger_violations = self.soft_constraints.is_violated(states)
        assert len(larger_violations.shape) == 1
        return larger_violations
    
    def check_violation(self, states: np.array):
        violations = self.constraints.is_violated(states)
        assert len(violations.shape) == 1
        return violations

    def _soft_constraint_values(self, states: np.array):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2

        if states.shape[0] > 1:
            return self.soft_constraints.get_value(states)  # (n, con_num)
        else:
            return self.soft_constraints.get_value(states).squeeze() # (con_num,)
    
    def _constraint_values(self, states: np.array):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2

        if states.shape[0] > 1:
            return self.constraints.get_value(states)  # (n, con_num)
        else:
            return self.constraints.get_value(states).squeeze() # (con_num,)

    def get_constraint_values(self, states):
        '''Compute the constraints values given states

        params:
            states shape: (n, dim_s) or (dim_s)
        return:
            constraint_value: shape (n, self.con_dim) or (self.con_dim,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2

        return np.squeeze(self.constraints.get_value(states))

if __name__ == "__main__":
    env = SafeInvertedPendulumEnv(
        task='move'
    )
    s = env.reset()
    print(s)
    for i in range(100):
        act = env.action_space.sample()
        s_p, r, done, info = env.step(act)
        print(act, s_p, r, done, info)
        print(env.check_done(np.tile(s_p, [2,1])))
        print(env.check_violation(np.tile(s_p, [2,1])))
        print("---------------------------------")

        if done:
            break