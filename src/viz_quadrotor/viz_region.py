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
    import scipy
    mytree = scipy.spatial.cKDTree(targets)
    dist, indexes = mytree.query(points_list, k=1)
    return indexes


class Vizer_set(object):
    def __init__(self,
                 cfg,
                 test_log_dir,
                 epoch,
                 bound=(-1.5, 1.5, 0.5, 1.5),
                 z_dot_list=[-1., 0., 1.]):
        # 1 Load params and models
        tester = Tester(cfg, test_log_dir, epoch)
        self.tester = tester
        self.z_dot_list = z_dot_list
        self.test_log_dir = test_log_dir

        # 2 Generate batch observations
        # 2.0 Generate ref trj
        QUADROTOR_CFG = config_eval.quadrotor_config
        TASK_INFO = QUADROTOR_CFG.task_info
        self.X_GOAL = _generate_ref_trj(
            traj_type=TASK_INFO["trajectory_type"],
            traj_length=QUADROTOR_CFG.episode_len_sec,
            num_cycles=TASK_INFO["num_cycles"],
            traj_plane=TASK_INFO["trajectory_plane"],
            position_offset=TASK_INFO["trajectory_position_offset"],
            scaling=TASK_INFO["trajectory_scale"],
            sample_time=1./QUADROTOR_CFG.ctrl_freq
        )  # shape: (epi_len_sec * ctrl_freq, 6) = (360, 6)

        # 2.1 Generate location obses
        x = np.linspace(bound[0], bound[1], 100)
        z = np.linspace(bound[2], bound[3], 100)
        X, Z = np.meshgrid(x, z)
        flatten_x = X.ravel()
        flatten_z = Z.ravel()
        batch_obses = np.zeros((len(flatten_x), self.X_GOAL.shape[1] * 2), dtype=np.float32)  # (100*100, 12)
        assert batch_obses.shape == (100*100, 12)
        batch_obses[:, 0] = flatten_x
        batch_obses[:, 2] = flatten_z
        self.X = X
        self.Z = Z

        # 2.2 Allocate ref point for each obs
        batch_location = batch_obses[:, [0, 2]]
        target_location = self.X_GOAL[:, [0, 2]]
        indexes = _find_closet_target(target_location, batch_location)
        batch_targets = self.X_GOAL[indexes, :]
        batch_obses[:, 6:] = batch_targets

        # 2.3 Copy batch obses to the num of z_dot_list
        self.batch_obses_list = []
        for z_dot in z_dot_list:
            obses = batch_obses.copy()
            obses[:, 3] = z_dot  # assign z_dot
            obses[:, 1] = batch_targets[:, 1]  # assign x_dot (same with target point)
            self.batch_obses_list.append(obses)
    
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
            for i, obses in enumerate(self.batch_obses_list):
                states = torch.from_numpy(obses).float().to(device)
                actions = self.tester.alg.solver.act(states, eval=True)
                qcs = self.tester.alg.solver.constraint_critic(states, actions)

                if self.tester.alg.solver.constrained_fcn == 'reachability':
                    qcs, _ = torch.max(qcs, dim=1)

                flatten_qcs = qcs.cpu().numpy()

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
                if len(metrics) == 1 and len(self.z_dot_list) == 1:
                    sub_ax = axes
                elif len(axes.shape) > 1:
                    sub_ax = axes[j][i]
                elif len(self.z_dot_list) > 1:
                    sub_ax = axes[i]
                elif len(metrics) > 1:
                    sub_ax = axes[j]

                ct = sub_ax.contourf(self.X, self.Z, data2plot[i],
                                     norm=norm,
                                     cmap='rainbow',
                                    #  levels=[-0.06, -0.04, -0.02, 0., 0.2, 0.6, 1.0, 1.2],  # CBF
                                    #  levels=[-2.4, -1.2, 0., 1.2, 2.4, 3.6, 4.8],  # RAC
                                    #  levels=[-1.2, -0.8, -0.4, 0., 0.4, 0.8, 1.2],  # SI
                                     )  
                                     
                ct_line = sub_ax.contour(self.X, self.Z, data2plot[i],
                                         levels=[0], colors='black',
                                         linewidths=2.5, linestyles='solid')
                sub_ax.clabel(ct_line, inline=True, fontsize=14, fmt=r'0',)

                x = np.linspace(-1.5, 1.5, 100)
                y1 = np.ones_like(x) * 1.5
                y2 = np.ones_like(x) * 0.5

                theta = np.linspace(0, 2*np.pi, 360)
                trj_x = np.cos(theta)
                trj_y = np.sin(theta) + 1
                line1, = sub_ax.plot(x, y1, color='black', linestyle = 'solid')
                line2, = sub_ax.plot(x, y2, color='black', linestyle = 'solid')
                line3, = sub_ax.plot(trj_x, trj_y, color='black', linestyle='dashed')

                sub_ax.set_yticks(np.linspace(0.5, 1.5, 3))
                ct_list.append(ct)
                sub_ax.set_title(r'$\dot{z}=$' + str(self.z_dot_list[i]))
                axes_list.append(sub_ax)

            # cax = add_right_cax(sub_ax, pad=0.01, width=0.02)
            plt.colorbar(ct_list[1], ax=axes_list,
                         shrink=0.8, pad=0.02)

        fig.supxlabel('x')
        fig.supylabel('z')
        plt.savefig(str(LOGS_DIR / self.test_log_dir / (metric + str(self.tester.alg.epochs_completed.item()) + '.png')), dpi=300)

def main():
    # step 1: load config and model
    cfg, test_log_dir, epochs = build_parser()

    # print(epochs)
    for i, epoch in enumerate(epochs[0]):
        if i == 0:
            vizer = Vizer_set(cfg, test_log_dir, epoch, bound=[-1.5, 1.5, 0, 2])
        else:
            vizer.tester.load_model(epoch)
        vizer.plot_region(['qc'])


if __name__ == '__main__':
    # Usage: in the command line, input the followings
    # $ python viz_region.py --run-dir <log_dir> --set env_name quadrotor --epoch <epoch_id, can be more than 1>
    main()