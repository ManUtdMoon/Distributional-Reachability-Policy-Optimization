from pathlib import Path
import sys
from turtle import colormode

PROJ_DIR = Path.cwd().parent.parent
sys.path.append(str(PROJ_DIR))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
import matplotlib.patches as patches
import matplotlib.font_manager as fm

import torch

from src.defaults import ROOT_DIR
from src.torch_util import device
from src.tester import Tester, build_parser


ROOT_DIR = Path(ROOT_DIR)
assert ROOT_DIR.is_dir(), ROOT_DIR
LOGS_DIR = ROOT_DIR / 'logs' / 'double_integrator'

fm.fontManager.addfont(str(PROJ_DIR / 'arial.ttf'))

params = {
    'font.family': 'Arial',
    'font.size': 15,  # or large,small
}
rcParams.update(params)

labelsize = 9
class Vizer_set(object):
    def __init__(self,
                 cfg,
                 test_log_dir,
                 epoch,
                 bound=(-5., 5., -5., 5.)):
        # 1 Load params and models
        tester = Tester(cfg, test_log_dir, epoch)
        self.tester = tester
        self.test_log_dir = test_log_dir

        # 2 Generate batch observations
        x1 = np.linspace(bound[0], bound[1], 101)
        x2 = np.linspace(bound[2], bound[3], 101)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        flatten_x1 = x1_grid.ravel()
        flatten_x2 = x2_grid.ravel()
        batch_obses = np.zeros((len(flatten_x1), 2), dtype=np.float32)  # (101*101, 2)
        assert batch_obses.shape == (101*101, 2)
        batch_obses[:, 0] = flatten_x1
        batch_obses[:, 1] = flatten_x2

        self.obses = batch_obses
        self.x1 = x1
        self.x2 = x2
        self.x1_grid = x1_grid
        self.x2_grid = x2_grid
        self.bound = bound
    
    @torch.no_grad()
    def plot_traj(self):
        plt.rcParams['figure.constrained_layout.use'] = True
        fig = plt.figure(figsize=(8, 4))

        # fig 1: traj
        ax1 = plt.subplot(131)
        # --------- get RL policy trajectoiry start --------- #
        traj = self._get_eval_traj()
        states = traj[0].get('states').cpu().numpy()
        actions = traj[0].get('actions').cpu().numpy()
        rl_x1 = states[:, 0]
        rl_x2 = states[:, 1]
        time_step_rl = np.arange(rl_x1.shape[0])
        rl_traj = ax1.scatter(rl_x1, rl_x2, s=5, c=time_step_rl, cmap='GnBu')
        # --------- get RL policy trajectoiry end --------- #

        # --------- get MPC trajectoiry start --------- #
        mpc_res = np.load('./mpc.npy', allow_pickle=True).item()
        # print(mpc_res)
        mpc_x = mpc_res['state']
        assert mpc_x.shape[1] == 2
        time_step_mpc = np.arange(mpc_x[:, 0].shape[0])
        mpc_traj = ax1.scatter(mpc_x[:, 0], mpc_x[:, 1], s=5, c=time_step_mpc, cmap='YlOrBr')
        # --------- get MPC trajectoiry end --------- #
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.set_aspect(1)
        ax1.tick_params(axis='x', labelsize=labelsize)
        ax1.tick_params(axis='y', labelsize=labelsize)

        cbar1 = plt.colorbar(rl_traj, shrink=0.6, pad=0.02)
        cbar1.ax.tick_params(labelsize=labelsize)
        cbar2 = plt.colorbar(mpc_traj, shrink=0.6, pad=0.02)
        cbar2.ax.tick_params(labelsize=labelsize)

        # fig 2: x1 traj
        ax2 = plt.subplot(322)
        ax2.plot(time_step_rl * 0.1, rl_x1)
        ax2.plot(time_step_mpc * 0.1, mpc_x[:, 0])
        ax2.set_ylim(-5, 5)
        ax2.tick_params(axis='x', labelsize=labelsize)
        ax2.tick_params(axis='y', labelsize=labelsize)
        ax2.set_ylabel(r'$x_1$', fontsize=labelsize)

        # fig 3: x2 traj
        ax3  = plt.subplot(324, sharex=ax2)
        ax3.plot(time_step_rl * 0.1, rl_x2)
        ax3.plot(time_step_mpc * 0.1, mpc_x[:, 1])
        ax3.set_ylim(-5, 5)
        ax3.set_ylabel(r'$x_2$', fontsize=labelsize)
        ax3.tick_params(axis='x', labelsize=labelsize)
        ax3.tick_params(axis='y', labelsize=labelsize)

        # fig 4: act traj
        ax4  = plt.subplot(326, sharex=ax2)
        ax4.plot(time_step_rl * 0.1 + 0.1, actions)
        ax4.plot(time_step_mpc[1:] * 0.1, mpc_res['action'])
        ax4.set_xlabel("times [s]", fontsize=labelsize)
        ax4.set_ylabel("action", fontsize=labelsize)
        ax4.tick_params(axis='x', labelsize=labelsize)
        ax4.tick_params(axis='y', labelsize=labelsize)

        ax1.set_xlabel(r'$x_1$', fontsize=labelsize)
        ax1.set_ylabel(r'$x_2$', fontsize=labelsize)
        plt.savefig(str(LOGS_DIR / self.test_log_dir / (str(self.tester.alg.epochs_completed.item()) + '.png')), dpi=300)

    @torch.no_grad()
    def _get_eval_traj(self):
        test_trajs, _ = self.tester.run_evaluation()
        return test_trajs


def main():
    # step 1: load config and model
    cfg, test_log_dir, epochs = build_parser()

    # print(epochs)
    for i, epoch in enumerate(epochs[0]):
        if i == 0:
            vizer = Vizer_set(cfg, test_log_dir, epoch, bound=[-5., 5., -5., 5.])
        else:
            vizer.tester.load_model(epoch)
        vizer.plot_traj()


if __name__ == '__main__':
    # Usage: in the command line, input the followings
    # $ python viz_trj_di.py --set env_name double_integrator --motivation traj --run-dir <log_dir> --epoch <epoch_id, can be more than 1>
    main()