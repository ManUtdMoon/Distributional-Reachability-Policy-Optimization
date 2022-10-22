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

import torch

from src.defaults import ROOT_DIR
from src.torch_util import device
from src.tester import Tester, build_parser


ROOT_DIR = Path(ROOT_DIR)
assert ROOT_DIR.is_dir(), ROOT_DIR
LOGS_DIR = ROOT_DIR / 'logs' / 'double_integrator'


params = {
    'font.family': 'Arial',
    'font.size': 15,  # or large,small
}
rcParams.update(params)


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
        fig, ax = plt.subplots(nrows=1, ncols=1,
                                 figsize=(4, 3),
                                 constrained_layout=True)
        
        # --------- get RL policy trajectoiry start --------- #
        traj = self._get_eval_traj()
        states = traj[0].get('states').cpu().numpy()
        rl_x1 = states[:, 0]
        rl_x2 = states[:, 1]
        time_step = np.arange(rl_x1.shape[0])
        rl_traj = ax.scatter(rl_x1, rl_x2, s=5, c=time_step, cmap='GnBu')
        # --------- get RL policy trajectoiry end --------- #

        # --------- get MPC trajectoiry start --------- #
        mpc_res = np.load('./mpc.npy', allow_pickle=True).item()
        # print(mpc_res)
        mpc_x = mpc_res['state']
        assert mpc_x.shape[1] == 2
        time_step = np.arange(mpc_x[:, 0].shape[0])
        mpc_traj = ax.scatter(mpc_x[:, 0], mpc_x[:, 1], s=5, c=time_step, cmap='YlOrBr')
        # --------- get MPC trajectoiry end --------- #
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        plt.colorbar(rl_traj, shrink=0.8, pad=0.02)
        plt.colorbar(mpc_traj, shrink=0.8, pad=0.02)
        fig.supxlabel(r'$x_1$')
        fig.supylabel(r'$x_2$')
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