from pathlib import Path
import sys
from collections import OrderedDict

import gym.spaces
import numpy as np

from safety_gym.envs.engine import Engine


def normalize_obs(pos, robot_pos, robot_mat):
    pos = np.concatenate((pos, np.zeros((pos.shape[0], 1))), axis=-1)
    vec = (pos - robot_pos) @ robot_mat
    x, y = vec[:, 0], vec[:, 1]
    z = x + 1j * y
    dist = np.abs(z)
    # dist = np.exp(-dist)
    angle = np.angle(z)
    return np.stack((dist, np.cos(angle), np.sin(angle)), axis=-1)


class SimpleEngine(Engine):
    DEFAULT = {
        **Engine.DEFAULT,
        'observe_hazards_pos': False,
    }

    def obs(self):
        self.sim.forward()
        obs = {}

        obs['accelerometer'] = self.world.get_sensor('accelerometer')[:2]
        obs['velocimeter'] = self.world.get_sensor('velocimeter')[:2]
        obs['gyro'] = self.world.get_sensor('gyro')[-1:]
        obs['magnetometer'] = self.world.get_sensor('magnetometer')[:2]
        if 'doggo.xml' in self.robot_base: self.extra_sensor_obs(obs)  # Must call after simplified sensors

        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        obs['goal_pos'] = normalize_obs(self.goal_pos[np.newaxis, :2], robot_pos, robot_mat)
        if self.observe_hazards_pos:
            obs['hazards_pos'] = normalize_obs(np.stack(self.hazards_pos)[:, :2], robot_pos, robot_mat)
        else:
            obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, None)

        flat_obs = np.zeros(self.obs_flat_size)
        for k, v in self.observation_layout.items():
            flat_obs[v] = obs[k].flat

        return flat_obs

    def build_observation_space(self):
        obs_space_dict = OrderedDict()

        obs_space_dict['goal_pos'] = gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32)

        obs_space_dict['accelerometer'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        obs_space_dict['velocimeter'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        obs_space_dict['gyro'] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        obs_space_dict['magnetometer'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        if 'doggo.xml' in self.robot_base:
            self.build_extra_sensor_observation_space(obs_space_dict)  # Must call after simplified sensors

        if self.observe_hazards_pos:
            obs_space_dict['hazards_pos'] = gym.spaces.Box(-np.inf, np.inf, (self.hazards_num, 3), dtype=np.float32)
        else:
            obs_space_dict['hazards_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)

        self.obs_space_dict = obs_space_dict
        self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        self.observation_layout = OrderedDict()

        offset = 0
        for k in self.obs_space_dict.keys():
            space = self.obs_space_dict[k]
            size = np.prod(space.shape)
            start, end = offset, offset + size
            self.observation_layout[k] = slice(start, end)
            offset += size
        assert offset == self.obs_flat_size

    def extra_sensor_obs(self, obs):
        from safety_gym.envs.engine import quat2mat
        for sensor in self.sensors_obs:
            if sensor in obs: continue
            obs[sensor] = self.world.get_sensor(sensor)
        for sensor in self.robot.hinge_vel_names:
            obs[sensor] = self.world.get_sensor(sensor)
        for sensor in self.robot.ballangvel_names:
            obs[sensor] = self.world.get_sensor(sensor)
        if self.sensors_angle_components:
            for sensor in self.robot.hinge_pos_names:
                theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
            for sensor in self.robot.ballquat_names:
                quat = self.world.get_sensor(sensor)
                obs[sensor] = quat2mat(quat)
        else:  # Otherwise read sensors directly
            for sensor in self.robot.hinge_pos_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballquat_names:
                obs[sensor] = self.world.get_sensor(sensor)

    def build_extra_sensor_observation_space(self, obs_space_dict):
        for sensor in self.sensors_obs:
            if sensor in obs_space_dict: continue
            dim = self.robot.sensor_dim[sensor]
            obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        for sensor in self.robot.hinge_vel_names:
            obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        for sensor in self.robot.ballangvel_names:
            obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        if self.sensors_angle_components:
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32)
        else:
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

    def step(self, action):
        action = np.array(action, copy=False)  # Cast to ndarray
        next_obs, reward, done, info = super(SimpleEngine, self).step(action)
        if 'cost_exception' in info:
            # Simulation exception
            # Example: MujocoException Got MuJoCo Warning: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 2.3700.
            assert 'cost' not in info and done
            assert not np.isnan(action).any()
            info['cost'] = info['cost_exception']
        return next_obs, reward, done, info


if __name__ == "__main__":
    PROJ_DIR = Path.cwd().parent.parent.parent
    sys.path.append(str(PROJ_DIR))

    from src.env.sg.config import point_goal_config

    env = SimpleEngine(point_goal_config)
    obs = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(info)
