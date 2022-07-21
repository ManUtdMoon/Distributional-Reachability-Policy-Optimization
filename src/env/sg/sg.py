import math
from copy import deepcopy
from pathlib import Path

import numpy as np

import gym
from gym import Wrapper

from src.env.sg.simple_safety_gym import SimpleEngine
from src.env.sg.config import *


class SafetyGymWrapper(Wrapper):
    def __init__(self, robot_type) -> None:
        env_cfg_dict = {
            'point': point_goal_config,
            'car': car_goal_config
        }
        env = SimpleEngine(env_cfg_dict[robot_type])
        super().__init__(env)

        self._max_episode_steps = self.num_steps

        self.con_dim = 1
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_flat_size + 2,), dtype=np.float32
        )  # the additional "2" is current constraint h(s) and isException
        self.info = None

    def augment_obs(self, obs):
        '''Augment the obs with a scalar representing the current constraint value.
        
        The constraint value cannot be computed with $s$ only because the information 
        about hazards is necessary. We can only access it with self.env

        params:
            obs
        return:
            obs_aug = (obs, h(s), isException)
        '''
        assert len(obs.shape) == 1, print(f"Obs has more than 1 dims, the dim is {obs.shape}")
        
        # start: compute hazards constraint values
        self.env.sim.forward()
        constraint_values = []
        for h_pos in self.env.hazards_pos:
            h_dist = self.env.dist_xy(h_pos)
            constraint_values.append(self.env.hazards_size - h_dist)
        constraint_value = np.max(constraint_values)
        # end: compute hazards constraint values

        isException = float(self.env.done)
        return np.concatenate([obs, [constraint_value, isException]])
    
    def step(self, action: np.array):
        new_info = dict(
            constraint_value=np.max(self.info['constraint_hazards']),
        )
        next_obs, rew, done, info = super().step(action)
        next_state = self.augment_obs(next_obs)
        new_info.update(dict(
            violation=(next_state[-2] > 0),
            **info
        ))
        self.info = new_info

        return next_state, rew, done, new_info

    def reset(self):
        obs = super().reset()
        state = self.augment_obs(obs)

        self.env.sim.forward()
        constraint_hazards = []
        for h_pos in self.env.hazards_pos:
            h_dist = self.env.dist_xy(h_pos)
            constraint_hazards.append(self.env.hazards_size - h_dist)
        
        self.info = dict(
            constraint_value=state[-2],
            constraint_hazards=constraint_hazards
        )

        return state

    def check_done(self, states: np.array):
        '''Compute whether exception happens

        params:
            states shape: (*, dim_s) where * can be any number of dimensions incl. none
        return:
            dones: shape (*,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return (states[..., -1] == 1)

    def check_violation(self, states: np.array):
        '''Compute whether the constraints are violated

        params:
            states shape: (*, dim_s)
        return:
            violations: shape (*,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return states[..., -2] > 0

    def get_constraint_values(self, states):
        '''Compute the constraints values given states

        params:
            states shape: (n, dim_s) or (dim_s)
        return:
            constraint_value: shape (n, self.con_dim) or (self.con_dim,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return states[..., -2]


if __name__ == '__main__':
    env = SafetyGymWrapper(
        robot_type='car',
    )
    np.set_printoptions(precision=5, suppress=True)
    s = env.reset()
    print(s)
    for i in range(100):
        act = env.action_space.sample()
        s_p, r, done, info = env.step(act)
        print(act, s_p, r, done, info)
        print(env.check_done(np.tile(s_p, [3, 2, 1])))
        print(env.check_violation(np.tile(s_p, [3, 2, 1])))
        print(env.get_constraint_values(np.tile(s_p, [3, 2, 1])))
        print(f"--------------step {i}---------------")

        if done:
            break