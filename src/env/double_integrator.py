from typing import Tuple

import gym
from gym.utils import seeding
import numpy as np



class DoubleIntegrator(gym.Env):
    def __init__(self, id=None):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.dt = 0.1
        self.id = id
        self.state = None
        self.seed()

        self.con_dim = 4
        self._max_episode_steps = 100
        self.done_on_out = (self.id is None)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self) -> np.ndarray:
        self.state = self.np_random.uniform(low=-5, high=5, size=2)
        if self.id is not None:
            self.state = np.array([-4.8, 4], dtype=np.float32)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        a = np.clip(action, self.action_space.low, self.action_space.high)[0]
        x1, x2 = self.state
        cons_vals = self.get_constraint_values(self.state)
        new_x1 = x1 + x2 * self.dt
        new_x2 = x2 + a * self.dt
        self.state[0] = new_x1
        self.state[1] = new_x2
        if self.done_on_out:
            done = self.check_done(self.state).item()
        else:
            if abs(self.state[0]) < 0.1 and abs(self.state[1]) < 0.1:
                done = True
            else:
                done = False
        info = dict(
            violation=self.check_violation(self.state).item(),
            constraint_value=cons_vals,
        )
        return self._get_obs(), self._get_reward(self.state, action), done, info

    def _get_obs(self):
        return np.copy(self.state)

    def _get_reward(self, state, action):
        x1, x2 = state
        rew_s = - abs(x1) - abs(x2)
        rew_a = - abs(action)
        return rew_s + 0.05 * rew_a
    
    def get_constraint_values(self, states):
        if len(states.shape) == 1:
            batched_states = states[np.newaxis, ...].copy()
        else:
            assert len(states.shape) >= 2
            batched_states = states.copy()

        x1s = batched_states[..., 0]
        x2s = batched_states[..., 1]
        batch_size = batched_states.shape[:-1]

        con_x1_lb = -x1s - 5.
        con_x1_ub = x1s - 5.
        con_x2_lb = -x2s - 5.
        con_x2_ub = x2s - 5.
        cons = np.array([con_x1_lb, con_x1_ub, con_x2_lb, con_x2_ub]).T.squeeze()
        assert cons.shape == (*batch_size, 4) or cons.shape == (4,)

        return cons

    def check_done(self, states):
        return np.logical_or(
            self.check_out(states),
            self.check_stable(states),
        )
    
    def check_stable(self, states):
        if len(states.shape) == 1:
            batched_states = states[np.newaxis, ...].copy()
        else:
            assert len(states.shape) >= 2
            batched_states = states.copy()

        x1s = batched_states[..., 0]
        x2s = batched_states[..., 1]
        batch_size = batched_states.shape[:-1]

        x1_stable = np.logical_and(x1s < 0.1, x1s > -0.1)
        x2_stable = np.logical_and(x2s < 0.1, x2s > -0.1)
        stables = np.logical_and(x1_stable, x2_stable)
        assert stables.shape == batch_size

        return stables

    def check_violation(self, states, margin=0.0):
        if len(states.shape) == 1:
            batched_states = states[np.newaxis, ...].copy()
        else:
            assert len(states.shape) >= 2
            batched_states = states.copy()

        x1s = batched_states[..., 0]
        x2s = batched_states[..., 1]
        batch_size = batched_states.shape[:-1]

        x1_violations = np.logical_or(x1s < -5. - margin, x1s > 5. + margin)
        x2_violations = np.logical_or(x2s < -5. - margin, x2s > 5. + margin)
        violations = np.logical_or(x1_violations, x2_violations)
        assert violations.shape == batch_size
        
        return violations

    def check_out(self, states):
        return self.check_violation(states, margin=1.0)

    def get_rewards(self, states, actions):
        assert len(states.shape) >= 2
        x1s = states[..., 0]
        x2s = states[..., 1]
        actions = actions.squeeze()
        rews = - np.abs(x1s) - np.abs(x2s) - 0.05 * np.abs(actions)
        batch_size = states.shape[:-1]
        assert rews.shape == batch_size == actions.shape, print(rews.shape, batch_size, actions.shape)
        return rews

    # def plot_map(self, ax):
    #     x1 = np.linspace(-5, 5, 101)
    #     x2 = np.linspace(-5, 5, 101)
    #     x1_grid, x2_grid = np.meshgrid(x1, x2)
    #     obs = np.stack((x1_grid, x2_grid), axis=2)

    #     x2_min = -np.sqrt(2 * (x1 + 5))
    #     x2_max = np.sqrt(2 * (5 - x1))
    #     ax.plot(x1, x2_min, color='k')
    #     ax.plot(x1, x2_max, color='k')

    #     feasible = (x2_grid >= x2_min) & (x2_grid <= x2_max)
    #     y_true = feasible * 0 + ~feasible * 1

    #     barrier = (x2_grid >= 0) * (x1_grid - 5 + x2_grid) + (x2_grid <= 0) * (-5 - x1_grid - x2_grid)

    #     return {
    #         'xs': x1_grid,
    #         'ys': x2_grid,
    #         'obs': obs,
    #         'y_true': y_true,
    #         'cbf': barrier,
    #         'x_label': r'$\mathrm{x_1}$',
    #         'y_label': r'$\mathrm{x_2}$',
    #     }

if __name__ == "__main__":
    env = DoubleIntegrator()
    s = env.reset()
    print(s)
    for i in range(50):
        print(f"----------- step {i} -------------")
        act = env.action_space.sample()
        s_p, r, done, info = env.step(act)
        print(act, s_p, r, done, info)
        print(env.check_done(np.tile(s_p, [2,1])))
        print(env.check_violation(np.tile(s_p, [2,1])))
        
        if done:
            break