import math
from copy import deepcopy
from pathlib import Path

import numpy as np

import gym
from gym import Wrapper

from src.env.sg.simple_safety_gym import SimpleEngine
from src.env.sg.config import *


class SafetyGymWrapper(Wrapper):
    def __init__(self, robot_type, id=None):
        env_cfg_dict = {
            'point': point_goal_config,
            'car': car_goal_config
        }
        env_cfg = deepcopy(env_cfg_dict[robot_type])
        # if id is None:  # for train env
        #     env_cfg['continue_goal'] = False
        # else:  # for eval env
        #     env_cfg['continue_goal'] = True
        env = SimpleEngine(env_cfg)
        super().__init__(env)

        self._max_episode_steps = self.num_steps

        self.con_dim = 1
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_flat_size + 1,), dtype=np.float32
        )  # the additional "2" is current constraint h(s) and isException
        self.info = None
        self.margin_scale = 0.9  # the closer to 1, the harder to done
        assert 0. < self.margin_scale < 1.

    def augment_obs(self, obs):
        '''Augment the obs with a scalar representing the current constraint value.
        
        The constraint value cannot be computed with $s$ only because the information 
        about hazards is necessary. We can only access it with self.env

        params:
            obs
        return:
            obs_aug = (obs, h(s))
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

        return np.concatenate([obs, [constraint_value]])

    def set_state_and_get_obs(self, layout, velocimeter, robot_rot):
        self.env.layout = layout
        self.world_config_dict = self.build_world_config()
        self.world_config_dict['robot_rot'] = robot_rot
        self.world.reset(build=False)
        self.world.rebuild(self.world_config_dict, state=False)
        obs = self.obs()
        obs[5:7] = velocimeter
        obs = self.augment_obs(obs)

        return obs
    
    def step(self, action: np.array):
        next_obs, rew, done, info = super().step(action)
        next_state = self.augment_obs(next_obs)
        new_info = dict(
            constraint_value=next_state[-1],
            violation=(next_state[-1] > 0),
            **info
        )
        self.info = new_info

        # if next_state[-1] >= self.margin_scale * self.env.hazards_size:
        #     done = True
        if (next_state[-1] > 0): done = True

        return next_state, rew, done, new_info

    def reset(self):
        obs = super().reset()
        state = self.augment_obs(obs)

        self.env.sim.forward()
        # constraint_hazards = []
        # for h_pos in self.env.hazards_pos:
        #     h_dist = self.env.dist_xy(h_pos)
        #     constraint_hazards.append(self.env.hazards_size - h_dist)
        
        # self.info = dict(
        #     constraint_value=state[-1],
        #     constraint_hazards=constraint_hazards
        # )

        return state

    def check_done(self, states: np.array):
        '''Compute whether soft constraint is violated or 
                           the agent reaches the goal region

        params:
            states shape: (*, dim_s) where * can be any number of dimensions incl. none
                [..., 0] is the exp(-dist) where dist is the distance to goal
                [..., -1] is the constraint function
        return:
            dones: shape (*,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return states[..., -1] > 0  # self.margin_scale * self.env.hazards_size

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

        return states[..., -1] > 0

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

        return states[..., -1]
    
    def check_goal_met(self, states):
        '''Compute whether the agents reach goal

        params:
            states shape: (*, dim_s)
        return:
            shape (*,) of True/False
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return states[..., 0] <= self.env.goal_size
    
    def get_reward(self, states: np.array, actions: np.array, next_states: np.array) -> np.array:
        '''Compute rewards of (s, a, s')
        In safety-gym, the rewards is only related to the dist_to_goal,
        i.e., the 0th of states

        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        if len(actions.shape) == 1:
            actions = actions[np.newaxis, ...]
        if len(next_states.shape) == 1:
            next_states = next_states[np.newaxis, ...]
        
        assert len(states.shape) >= 2 and (len(states.shape) == len(next_states.shape) == len(actions.shape))

        batch_size = states.shape[:-1]
        rewards = states[..., 0] - next_states[..., 0]
        assert rewards.shape == batch_size

        return rewards


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