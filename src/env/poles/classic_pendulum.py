from gym import register
import gym.spaces as spaces
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np

class SafeClassicPendulum(PendulumEnv):
    def __init__(self, 
        init_state, 
        threshold, 
        goal_state=(0., 0.), 
        max_torque=2.0, 
        obs_type='state', 
        task='upright', 
        **kwargs):
        
        self.init_state = np.array(init_state, dtype=np.float32).squeeze()
        self.goal_state = np.array(goal_state, dtype=np.float32).squeeze()
        self.threshold = threshold
        self.obs_type = obs_type
        self.task = task
        super().__init__(**kwargs)

        if obs_type == 'state':
            high = np.array([np.pi / 2, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        elif obs_type == 'observation':
            high = np.array([1, 1, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        else:
            assert 0

        self.max_torque = max_torque
        self.action_space = spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)
        self._max_episode_steps = 200

    def _get_obs(self):
        th, thdot = self.state
        if self.obs_type == 'state':
            return np.array([angle_normalize(th), thdot], dtype=np.float32)
        else:
            return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def reset(self):
        self.state = self.init_state
        self.last_u = None
        self.episode_unsafe = False
        return self._get_obs()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        # costs = (angle_normalize(th) - self.goal_state[0]) ** 2 + \
        #     0.1 * (thdot - self.goal_state[1]) ** 2  # + 0.001 * (u ** 2)
        costs = (angle_normalize(th) - self.goal_state[0]) ** 2

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot], np.float32)
        constraint_value = max(newth - self.threshold, -self.threshold-newth)
        violation = not (constraint_value <= 0.)
        info = dict(
            violation=violation,
            constraint_value=constraint_value
        )
        return self._get_obs(), -costs, False, info
    
    def check_done(self, states: np.array):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2
        return np.zeros((states.shape[0],), dtype=np.bool_)

    def check_violation(self, states: np.array):
        violations = np.logical_not(self._constraint_values(states) <= 0.)
        assert len(violations.shape) == 1
        return violations
    
    def _constraint_values(self, states: np.array):
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) == 2
        ths = states[:, 0]

        return np.maximum(ths - self.threshold, -self.threshold - ths)


if __name__ == '__main__':
    env = SafeClassicPendulum(
        init_state=[-0.3, -0.9],
        threshold=np.pi/2
    )
    state = env.reset()
    print(state)
    for i in range(10):
        act = env.action_space.sample()
        s_p, r, done, info = env.step(act)
        print(act, s_p, r, done, info)
        print(env.check_done(np.tile(s_p, [2,1])))
        print(env.check_violation(np.tile(s_p, [2,1])))
        print("---------------------------------")
    