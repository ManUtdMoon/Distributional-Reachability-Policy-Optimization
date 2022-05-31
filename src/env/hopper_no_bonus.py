import numpy as np
from gym.envs.mujoco.hopper import HopperEnv as GymHopperEnv

from .mujoco_wrapper import MujocoWrapper


class HopperEnv(GymHopperEnv, MujocoWrapper):
    @staticmethod
    def done(states):
        heights, angs = states[:,0], states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))

    def qposvel_from_obs(self, obs):
        qpos = np.zeros(6)
        qpos[1:] = obs[:5]
        qvel = obs[5:]
        return qpos, qvel


class HopperNoBonusEnv(HopperEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        reward -= 1     # subtract out alive bonus
        info['violation'] = done
        return next_state, reward, done, info

    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        heights, angs = states[:,0], states[:,1]
        return ~(np.isfinite(states).all(axis=1) & (np.abs(states[:,1:]) < 100).all(axis=1) & (heights > .7) & (np.abs(angs) < .2))