#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment with surrounding vehicles constraint
#  Update: 2022-11-20, Yujie Yang: create environment

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np

from src.env.tracking.pyth_veh3dofconti_data import SimuVeh3dofconti, angle_normalize

'''ref path and u combination
0: sine ref + sine u
1: sine ref + constant u (focus)
2: double lane ref + sine u
3: double lane ref + constant u (focus)
4: triangle ref + sine u
5: triangle ref + constant u
6: circle ref + sine u
7: circle ref + constant u
'''

@dataclass
class SurrVehicleData:
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    # front wheel angle
    delta: float = 0.0
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1

    def step(self):
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)


class SimuVeh3dofcontiSurrCstr2(SimuVeh3dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        surr_veh_num: int = 4,
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        ref_num: Optional[int] = None,
        id: Optional[int] = None,
        render: Optional[bool] = False,
        **kwargs: Any,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ego_obs_dim + ref_obs_dim * pre_horizon + surr_veh_num * 4,),
            dtype=np.float32,
        )  # 1: ego_phi, necessary for constraint calculation
        self.surr_veh_num = surr_veh_num
        self.surr_vehs: List[SurrVehicleData] = None
        self.surr_state = np.zeros((surr_veh_num, 5), dtype=np.float32)
        self.veh_length = veh_length
        self.veh_width = veh_width
        self.info_dict.update(
            {
                "surr_state": {"shape": (surr_veh_num, 5), "dtype": np.float32},
                "constraint": {"shape": (1,), "dtype": np.float32},
            }
        )

        self.ref_num = ref_num
        self.surr_vehs_start_dim = ego_obs_dim + 1 + ref_obs_dim * pre_horizon

        # ----- drpo-related -----
        self.con_dim = 1
        self._max_episode_steps = self.max_episode_steps
        self.done_on_violation = False
        self._id = id
        self.is_render = render

    def judge_done(self) -> bool:
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]

        # use distance in ego coordinates to judge done
        cos_tf = np.cos(-phi)
        sin_tf = np.sin(-phi)
        ref_x_tf = (ref_x - x) * cos_tf - (ref_y - y) * sin_tf
        ref_y_tf = (ref_x - x) * sin_tf + (ref_y - y) * cos_tf
        ref_phi_tf = angle_normalize(ref_phi - phi)

        done = (np.abs(ref_x_tf) > 50) | (np.abs(ref_y_tf) > 20) | (np.abs(ref_phi_tf) > np.pi)
        return done
    
    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        return_info = False,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if self._id is not None:
            ref_time = 0.0
            init_state = np.zeros(6, dtype=np.float32)
            init_state[3] = -1.0

        super().reset(init_state, ref_time, self.ref_num, **kwargs)

        surr_x0, surr_y0 = self.ref_points[0, :2]
        if self.path_num == 3:
            # circle path
            surr_phi = self.ref_points[0, 2]
            surr_delta = -np.arctan2(SurrVehicleData.l, self.ref_traj.ref_trajs[3].r)
        else:
            surr_phi = 0.0
            surr_delta = 0.0

        self.surr_vehs = []
        for _ in range(self.surr_veh_num):
            # avoid ego vehicle
            if self._id is None:
                while True:
                    # TODO: sample position according to reference trajectory
                    delta_lon = 10 * self.np_random.uniform(-1, 1)
                    delta_lat = 5 * self.np_random.uniform(-1, 1)
                    if abs(delta_lon) > 7 or abs(delta_lat) > 3:
                        break
                surr_u = 5 + self.np_random.uniform(-1, 1)
            else: # for evaluation and testing
                # TODO: design specific position for surr
                delta_lon = 5 # 8 for sine
                delta_lat = 3.5
                surr_u = 5 # 4.5 for sine
                print(f"surr {_}: d_lon: {delta_lon}, d_lat: {delta_lat}, u: {surr_u}")
            surr_x = (
                surr_x0 + delta_lon * np.cos(surr_phi) - delta_lat * np.sin(surr_phi)
            )
            surr_y = (
                surr_y0 + delta_lon * np.sin(surr_phi) + delta_lat * np.cos(surr_phi)
            )

            self.surr_vehs.append(
                SurrVehicleData(
                    x=surr_x,
                    y=surr_y,
                    phi=surr_phi,
                    u=surr_u,
                    delta=surr_delta,
                    dt=self.dt,
                )
            )
        self.update_surr_state()

        if return_info:
            return self.get_obs(), self.info
        else:
            return self.get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, _ = super().step(action)

        for surr_veh in self.surr_vehs:
            surr_veh.step()
        self.update_surr_state()

        info = self.info
        if self.is_render:
            info.update({
                "img": self.render(mode="rgb_array")
            })

        return self.get_obs(), reward, done.item(), info

    def update_surr_state(self):
        for i, surr_veh in enumerate(self.surr_vehs):
            self.surr_state[i] = np.array(
                [surr_veh.x, surr_veh.y, surr_veh.phi, surr_veh.u, surr_veh.delta],
                dtype=np.float32,
            )

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        ego_obs = obs[:6]
        ref_obs = obs[6:]
        ego_phi_abs = self.state[2]
        surr_obs = self.surr_state[:, :4] - self.state[np.newaxis, :4]
        return np.concatenate((ego_obs, ref_obs, surr_obs.flatten()))

    def get_constraint(self) -> float:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width

        x, y, phi = self.state[:3]
        ego_center = np.array(
            [
                [x + d * np.cos(phi), y + d * np.sin(phi)],
                [x - d * np.cos(phi), y - d * np.sin(phi)],
            ],
            dtype=np.float32,
        )

        surr_x = self.surr_state[:, 0]
        surr_y = self.surr_state[:, 1]
        surr_phi = self.surr_state[:, 2]
        surr_center = np.stack(
            (
                np.stack(
                    ((surr_x + d * np.cos(surr_phi)), surr_y + d * np.sin(surr_phi)),
                    axis=1,
                ),
                np.stack(
                    ((surr_x - d * np.cos(surr_phi)), surr_y - d * np.sin(surr_phi)),
                    axis=1,
                ),
            ),
            axis=1,
        )

        min_dist = np.inf
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = np.linalg.norm(
                    ego_center[np.newaxis, i] - surr_center[:, j], axis=1
                )
                min_dist = min(min_dist, np.min(dist))
        return 2 * r - min_dist

    @property
    def info(self):
        info = super().info
        info.update(
            {"surr_state": self.surr_state.copy(), "constraint_value": self.get_constraint(),}
        )
        info.update(
            {"violation": (info["constraint_value"] > 0.).item()}
        )
        return info

    def _render(self, ax):
        super()._render(ax, self.veh_length, self.veh_width)
        import matplotlib.patches as pc

        # draw surrounding vehicles
        for i in range(self.surr_veh_num):
            surr_x, surr_y, surr_phi = self.surr_state[i, :3]
            ax.add_patch(pc.Rectangle(
                (surr_x - self.veh_length / 2, surr_y - self.veh_width / 2), self.veh_length, self.veh_width, surr_phi * 180 / np.pi,
                facecolor='w', edgecolor='k', zorder=1))
    
    # ----- below is drpo-related methods, all batched -----

    def check_done(self, states: np.ndarray) -> np.ndarray:
        if states.ndim == 1:
            states = states[np.newaxis, ...]
        assert states.ndim == 2

        error_x = np.abs(states[:, 0])
        error_y = np.abs(states[:, 1])
        error_phi = np.abs(states[:, 2])
        done = np.logical_or(
            np.logical_or(error_x > 5, error_y > 2), error_phi > np.pi
        )
        return np.squeeze(done)

    def check_violation(self, states: np.ndarray) -> np.ndarray:
        if states.ndim == 1:
            states = states[np.newaxis, ...]
        assert states.ndim == 2

        return np.squeeze(self.get_constraint_values(states) > 0)

    def get_constraint_values(self, states: np.ndarray) -> np.ndarray:
        if states.ndim == 1:
            states = states[np.newaxis, ...]
        assert states.ndim == 2

        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width
        ego_center = np.array([[d, 0], [-d, 0]], dtype=np.float32)  # ego-coord

        # get the coordinates of circle centers of surrounding 
        # vehicles in ego-coord, shape: (batch_size, surr_veh_num, 2, 2)
        # the dim of surr_veh starts from self.surr_vehs_start_dim, every 4
        # dims is a surr_veh, compute the center needs 0:3, i.e., x, y, phi (all relative)
        phis = states[:, 6]
        cos_phis = np.expand_dims(np.cos(phis), axis=-1) # shape: (batch_size, 1)
        sin_phis = np.expand_dims(np.sin(phis), axis=-1) # shape: (batch_size, 1)
        surrs_rel_earth = states[:, self.surr_vehs_start_dim:].reshape(-1, self.surr_veh_num, 4)
        surrs_xs_rel_earth = surrs_rel_earth[:, :, 0] # shape: (batch_size, surr_veh_num)
        surrs_ys_rel_earth = surrs_rel_earth[:, :, 1]
        surrs_phis_rel = surrs_rel_earth[:, :, 2]

        surrs_xs_rel_ego = surrs_xs_rel_earth * cos_phis + surrs_ys_rel_earth * sin_phis
        surrs_ys_rel_ego = -surrs_xs_rel_earth * sin_phis + surrs_ys_rel_earth * cos_phis
        assert surrs_xs_rel_ego.shape[-1] == self.surr_veh_num

        surrs_centers_rel_ego = np.stack(
            (
                np.stack(
                    (
                        surrs_xs_rel_ego + d * np.cos(surrs_phis_rel),
                        surrs_ys_rel_ego + d * np.sin(surrs_phis_rel),
                    ),
                    axis=2,
                ),
                np.stack(
                    (
                        surrs_xs_rel_ego - d * np.cos(surrs_phis_rel),
                        surrs_ys_rel_ego - d * np.sin(surrs_phis_rel),
                    ),
                    axis=2,
                ),
            ),
            axis=2,
        )  # shape: (batch_size, surr_veh_num, 2, 2), 2 * (x, y)

        # increase the dim of ego_center to match the shape of surr_centers
        ego_center = ego_center[np.newaxis, np.newaxis, ...]

        # compute the distance between ego_center and surr_center, 
        # shape (batch_size, surr_veh_num)
        d1 = np.linalg.norm(ego_center[..., 0, :] - surrs_centers_rel_ego[..., 0, :], axis=-1)
        d2 = np.linalg.norm(ego_center[..., 0, :] - surrs_centers_rel_ego[..., 1, :], axis=-1)
        d3 = np.linalg.norm(ego_center[..., 1, :] - surrs_centers_rel_ego[..., 0, :], axis=-1)
        d4 = np.linalg.norm(ego_center[..., 1, :] - surrs_centers_rel_ego[..., 1, :], axis=-1)
        # get the nearest/smallest distance among d1~4
        min_dist = np.min(
            np.min(
                np.stack((d1, d2, d3, d4), axis=1),
                axis=-1
            ),  # shape (batch_size, surr_veh_num)
            axis=-1
        )  # shape (batch_size,)
            
        # the constraint value is 2 * r - min_dist, shape (batch_size,) or ()
        return np.squeeze(2 * r - min_dist)

def env_creator(**kwargs):
    return SimuVeh3dofcontiSurrCstr2(**kwargs)

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    env = SimuVeh3dofcontiSurrCstr2(
        ref_num=1,
    )
    s = env.reset()
    for i in range(100):
        print(f"-------------- {i} -------------------")
        act = env.action_space.sample()
        s_p, r, done, info = env.step(act)
        # print(f"s_p: {s_p}")
        # print(info)
        print(env.check_done(np.tile(s_p, [2,1])))
        print(env.check_violation(np.tile(s_p, [2,1])))

        assert np.isclose(info['constraint_value'], env.get_constraint_values(s_p), atol=1e-4), \
            print(f"info: {info['constraint_value']}, get_con_vals: {env.get_constraint_values(s_p)}")
        if done:
            break