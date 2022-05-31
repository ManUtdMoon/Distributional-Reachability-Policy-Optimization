import pdb

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv as GymHalfCheetahEnv
from .mujoco_wrapper import MujocoWrapper

class HalfCheetahEnv(GymHalfCheetahEnv, MujocoWrapper):
    @staticmethod
    def done(states):
        return np.zeros(len(states), dtype=bool)

    def qposvel_from_obs(self, obs):
        qpos = np.zeros(9)
        qpos[1:] = obs[:8]
        qvel = obs[8:]
        return qpos, qvel


class CheetahFlipTestEnv(HalfCheetahEnv):
    def check_termination(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name_set = set()
            name_set.add(self.model.geom_names[contact.geom1])
            name_set.add(self.model.geom_names[contact.geom2])
            if 'floor' in name_set and 'head' in name_set:
                return True
        return False


_flip_test_env = CheetahFlipTestEnv()

class CheetahNoFlipEnv(HalfCheetahEnv):
    def step(self, action):
        next_state, reward, _, info = super().step(action)
        _flip_test_env.set_state_from_obs(next_state)
        info['violation'] = _flip_test_env.check_termination() #self.check_termination()
        return next_state, reward, False, info

    def check_done(self, states):
        return np.zeros(len(states), dtype=bool)

    def check_violation(self, states):
        violations = []
        for state in states:
            _flip_test_env.set_state_from_obs(state)
            violations.append(_flip_test_env.check_termination())
        return np.array(violations)