from math import pi
from pathlib import Path
import sys

PROJ_DIR = Path.cwd().parent.parent
sys.path.append(str(PROJ_DIR))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams

import torch

from src.defaults import ROOT_DIR
from src.torch_util import device
from src.tester import Tester, build_parser
from src.env.quadrotor.quadrotor import config_eval
from src.env.sg.sg import SafetyGymWrapper


ROOT_DIR = Path(ROOT_DIR)
assert str(ROOT_DIR) == str(PROJ_DIR)
assert ROOT_DIR.is_dir(), ROOT_DIR
LOGS_DIR = ROOT_DIR / 'logs' / 'quadrotor'


params={'font.family': 'Arial',
        # 'font.serif': 'Times New Roman',
        # 'font.style': 'italic',
        # 'font.weight': 'normal', #or 'blod'
        'font.size': 15,  # or large,small
        }
rcParams.update(params)


def _generate_ref_trj(
    traj_type="circle",
    traj_length=6.0,
    num_cycles=1,
    traj_plane="xz",
    position_offset=np.array([0, 0]),
    scaling=1.0,
    sample_time=0.01
):
    def _circle(
        t,
        traj_period,
        scaling
    ):
        """Computes the coordinates of a circle trajectory at time t.
        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.
        Returns:
            float: The position in the first coordinate.
            float: The position in the second coordinate.
            float: The velocity in the first coordinate.
            float: The velocity in the second coordinate.
        """
        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.cos(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t)
        coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
        coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _get_coordinates(
        t,
        traj_type,
        traj_period,
        coord_index_a,
        coord_index_b,
        position_offset_a,
        position_offset_b,
        scaling
    ):
        """Computes the coordinates of a specified trajectory at time t.
        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_period (float): The period of the trajectory in seconds.
            coord_index_a (int): The index of the first coordinate of the trajectory plane.
            coord_index_b (int): The index of the second coordinate of the trajectory plane.
            position_offset_a (float): The offset in the first coordinate of the trajectory plane.
            position_offset_b (float): The offset in the second coordinate of the trajectory plane.
            scaling (float, optional): Scaling factor for the trajectory.
        Returns:
            ndarray: The position in x, y, z, at time t.
            ndarray: The velocity in x, y, z, at time t.
        """
        # Get coordinates for the trajectory chosen.
        if traj_type == "circle":
            coords_a, coords_b, coords_a_dot, coords_b_dot = _circle(
                t, traj_period, scaling)
        else:
            raise NotImplementedError("Unknown shape of trajectory")
        # Initialize position and velocity references.
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        # Set position and velocity references based on the plane of the trajectory chosen.
        pos_ref[coord_index_a] = coords_a + position_offset_a
        vel_ref[coord_index_a] = coords_a_dot
        pos_ref[coord_index_b] = coords_b + position_offset_b
        vel_ref[coord_index_b] = coords_b_dot
        return pos_ref, vel_ref

    # Get trajectory type.
    valid_traj_type = ["circle"]  # "square", "figure8"
    if traj_type not in valid_traj_type:
        raise ValueError("Trajectory type should be one of [circle, square, figure8].")
    traj_period = traj_length / num_cycles
    direction_list = ["x", "y", "z"]
    # Get coordinates indexes.
    if traj_plane[0] in direction_list and traj_plane[
        1] in direction_list and traj_plane[0] != traj_plane[1]:
        coord_index_a = direction_list.index(traj_plane[0])
        coord_index_b = direction_list.index(traj_plane[1])
    else:
        raise ValueError("Trajectory plane should be in form of ab, where a and b can be {x, y, z}.")
    # Generate time stamps.
    times = np.arange(0, traj_length, sample_time)
    pos_ref_traj = np.zeros((len(times), 3))
    vel_ref_traj = np.zeros((len(times), 3))
    # Compute trajectory points.
    for t in enumerate(times):
        pos_ref_traj[t[0]], vel_ref_traj[t[0]] = _get_coordinates(
            t[1],
            traj_type,
            traj_period,
            coord_index_a,
            coord_index_b,
            position_offset[0],
            position_offset[1],
            scaling
        )

    return np.vstack([
        pos_ref_traj[:, 0],
        vel_ref_traj[:, 0],
        pos_ref_traj[:, 2],
        vel_ref_traj[:, 2],
        np.zeros(pos_ref_traj.shape[0]),
        np.zeros(vel_ref_traj.shape[0])
    ]).transpose()


def _find_closet_target(targets, points_list):
    import scipy.spatial as spatial
    mytree = spatial.cKDTree(targets)
    dist, indexes = mytree.query(points_list, k=1)
    return indexes


class Vizer_set(object):
    def __init__(self,
                 cfg,
                 test_log_dir,
                 epoch,
                 bound=(-2.0, 2.0, -2.0, 2.0),
                 v_x_phi_list=[[1.0, pi] ,[1.0, pi/2], [1.0, 0]]):
        # 1 Load params and models
        tester = Tester(cfg, test_log_dir, epoch)
        self.tester = tester
        self.v_x_phi_list = v_x_phi_list
        self.test_log_dir = test_log_dir

        # 2 Generate batch observations

        # 2.0 load obs
        grid = np.load('grid.npz')
        obs = grid['obs']
        self.batch_obses_list = [obs[0], obs[1], obs[2]]
        n = grid['n']
        # 2.0 Generate env
        env_for_obs = SafetyGymWrapper(robot_type='car')
        env_for_obs.reset()
        self.layout = { 
        'robot': np.array([0.0, 0.0]), 
        'goal': np.array([-1.0, 0.7]), 
        'hazard0': np.array([-1.5, 1.5]), 
        'hazard1': np.array([-0.9, -0.2]), 
        'hazard2': np.array([1.2, 1.6]), 
        'hazard3': np.array([1.2, -1.2])
        }

        # 2.1 Generate location
        x = np.linspace(bound[0], bound[1], n)
        y = np.linspace(bound[2], bound[3], n)
        X, Y = np.meshgrid(x, y)
        flatten_x = X.ravel()
        flatten_y = Y.ravel()
        batch_obses = np.zeros((len(flatten_x),27), dtype=np.float32)  # (20*20, 12)
        assert batch_obses.shape == (n*n, 27)
        self.X = X
        self.Y = Y

    @torch.no_grad()
    def plot_region(self, metrics):
        def add_right_cax(ax, pad, width):
            '''
            在一个ax右边追加与之等高的cax.
            pad是cax与ax的间距.
            width是cax的宽度.
            '''
            axpos = ax.get_position()
            caxpos = mpl.transforms.Bbox.from_extents(
                axpos.x1 + pad,
                axpos.y0,
                axpos.x1 + pad + width,
                axpos.y1
            )
            cax = ax.figure.add_axes(caxpos)

            return cax

        for metric in metrics:
            assert metric in ['qc']

        fig, axes = plt.subplots(nrows=len(metrics), ncols=len(self.batch_obses_list),
                                 figsize=(12, 3),
                                 constrained_layout=True)
        # axes.set_position([0.1, 0.1, 0.9, 0.9])
        # fig_aux = plt.figure()

        for j, metric in enumerate(metrics):
            axes_list = []
            min_val_list = []
            max_val_list = []
            data2plot = []
            ct_list = []
            print("self.batch_obses_list", len(self.batch_obses_list))
            for i, obses in enumerate(self.batch_obses_list):
                states = torch.from_numpy(obses).float().to(device)
                states[:, -1] = states[:, -1]
                actions = self.tester.alg.solver.act(states, eval=True)
                qcs = self.tester.alg.solver.constraint_critic(
                    states, actions, 
                    uncertainty=self.tester.alg.solver.distributional_qc
                )
                # if self.tester.alg.solver.constrained_fcn == 'reachability':
                #     qcs, _ = torch.max(states, dim=1)

                flatten_qcs = qcs.cpu().numpy()

                # print("states", states)
                # qcs = states[:, 0:1]
                # flatten_qcs = qcs.cpu().numpy()
                NAME2VALUE = dict(zip(['qc'], [flatten_qcs]))
                val = NAME2VALUE[metric].reshape(self.X.shape)

                min_val_list.append(np.min(val))
                max_val_list.append(np.max(val))
                data2plot.append(val)

            min_val = np.min(min_val_list)
            max_val = np.max(max_val_list)
            min_idx = np.argmin(min_val_list)
            max_idx = np.argmax(max_val_list)
            norm = colors.Normalize(vmin=min_val, vmax=max_val)

            for i in range(len(data2plot)):
                if len(metrics) == 1 and len(self.v_x_phi_list) == 1:
                    sub_ax = axes
                elif len(axes.shape) > 1:
                    sub_ax = axes[j][i]
                elif len(self.v_x_phi_list) > 1:
                    sub_ax = axes[i]
                elif len(metrics) > 1:
                    sub_ax = axes[j]
                sub_ax.set_aspect(1)

                ct = sub_ax.contourf(self.X, self.Y, data2plot[i],
                                     norm=norm,
                                     cmap='rainbow',
                                    #  levels=[-0.06, -0.04, -0.02, 0., 0.2, 0.6, 1.0, 1.2],  # CBF
                                    #  levels=[-2.4, -1.2, 0., 1.2, 2.4, 3.6, 4.8],  # RAC
                                    #  levels=[-1.2, -0.8, -0.4, 0., 0.4, 0.8, 1.2],  # SI
                                     )  

                ct_line = sub_ax.contour(self.X, self.Y, data2plot[i],
                                         levels=[0], colors='black',
                                         linewidths=1.0, linestyles='solid')
                # sub_ax.clabel(ct_line, inline=True, fontsize=14, fmt=r'0',)

                theta = np.linspace(0, 2*np.pi, 360)
                hazard_size = 0.15
                hazard0_x = hazard_size * np.cos(theta) + self.layout['hazard0'][0]
                hazard0_y = hazard_size * np.sin(theta) + self.layout['hazard0'][1]
                hazard1_x = hazard_size * np.cos(theta) + self.layout['hazard1'][0]
                hazard1_y = hazard_size * np.sin(theta) + self.layout['hazard1'][1]
                hazard2_x = hazard_size * np.cos(theta) + self.layout['hazard2'][0]
                hazard2_y = hazard_size * np.sin(theta) + self.layout['hazard2'][1]
                hazard3_x = hazard_size * np.cos(theta) + self.layout['hazard3'][0]
                hazard3_y = hazard_size * np.sin(theta) + self.layout['hazard3'][1]
                goal_x =  0.3 * np.cos(theta) + self.layout['goal'][0]
                goal_y =  0.3 * np.sin(theta) + self.layout['goal'][1]

                sub_ax.plot(hazard0_x, hazard0_y, color='black', linestyle='dashed', linewidth=0.5)
                sub_ax.plot(hazard1_x, hazard1_y, color='black', linestyle='dashed', linewidth=0.5)
                sub_ax.plot(hazard2_x, hazard2_y, color='black', linestyle='dashed', linewidth=0.5)
                sub_ax.plot(hazard3_x, hazard3_y, color='black', linestyle='dashed', linewidth=0.5)
                sub_ax.plot(goal_x, goal_y, color='black', linestyle='solid', linewidth=0.5)

                sub_ax.set_yticks(np.linspace(-2.0, 2.0, 3))
                ct_list.append(ct)
                name_list = ["left", "up", "right"]
                sub_ax.set_title(name_list[i])
                axes_list.append(sub_ax)

            # cax = add_right_cax(sub_ax, pad=0.01, width=0.02)
            plt.colorbar(ct_list[1], ax=axes_list,
                         shrink=0.8, pad=0.02)

        fig.supxlabel('x')
        fig.supylabel('y')
        plt.savefig(str(LOGS_DIR / self.test_log_dir / (metric + str(self.tester.alg.epochs_completed.item()) + '.png')), dpi=300)

def main():
    # step 1: load config and model
    cfg, test_log_dir, epochs = build_parser()

    # print(epochs)
    for i, epoch in enumerate(epochs[0]):
        if i == 0:
            vizer = Vizer_set(cfg, test_log_dir, epoch, bound=[-2.0, 2.0, -2.0, 2.0])
        else:
            vizer.tester.load_model(epoch)
        vizer.plot_region(['qc'])

def test():
    np.set_printoptions(precision=5, suppress=True)
    cfg, test_log_dir, epochs = build_parser()
    # 1 Load params and models
    tester = Tester(cfg, test_log_dir, epochs[0][0])

    # 2 Generate batch observations
    # 2.0 Generate env
    env_for_obs = SafetyGymWrapper(robot_type='car')
    env_for_obs.reset()
    layout = { 
    'robot': np.array([0.0, 0.0]), 
    'goal': np.array([-1.0, 0.7]), 
    'hazard0': np.array([-1.5, 1.5]), 
    'hazard1': np.array([-0.9, -0.2]), 
    'hazard2': np.array([1.2, 1.6]), 
    'hazard3': np.array([1.2, -1.2])
    }
        # 2.2 convert state to obs
    def state2obs(env, basic_layout, x, y, phi, v_x, v_y):
        basic_layout['robot'] = np.array([x, y])
        velocimeter = [v_x, v_y]
        robot_rot = phi
        obs = env.set_state_and_get_obs(basic_layout, velocimeter, robot_rot)
        return obs

    robot_xy = [1.3, -1.6]
    velocimeter = [1.0, 0.0]
    robot_rot = pi * 0.0
    obs = state2obs(env=env_for_obs, basic_layout=layout, x=robot_xy[0], y=robot_xy[1], phi=robot_rot, v_x=velocimeter[0], v_y=velocimeter[1])
    # obs[0:3] = np.array([0.90554, 0.11043, 0.99388])
    print(obs)

    states = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    actions = tester.alg.solver.act(states, eval=True)
    qcs = tester.alg.solver.constraint_critic(states, actions)
    # if tester.alg.solver.constrained_fcn == 'reachability':
    #     qcs, _ = torch.max(qcs, dim=1)
    print("qcs", qcs)



if __name__ == '__main__':
    # Usage: in the command line, input the followings
    # $ python viz_region.py --run-dir <log_dir> --set env_name safetygym-car --epoch <epoch_id, can be more than 1>
    main()