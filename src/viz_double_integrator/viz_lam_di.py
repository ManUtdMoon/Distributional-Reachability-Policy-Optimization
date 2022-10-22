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
    def plot_region(self, metrics):

        for metric in metrics:
            assert metric in ['lam']

        fig, axes = plt.subplots(nrows=len(metrics), ncols=1,
                                 figsize=(4, 3),
                                 constrained_layout=True)

        for j, metric in enumerate(metrics):
            axes_list = []
            min_val_list = []
            max_val_list = []
            data2plot = []
            ct_list = []
            
            states = torch.from_numpy(self.obses).float().to(device)
            actions = self.tester.alg.solver.actor.act(states, eval=True)
            distributional_qc = self.tester.alg.solver.distributional_qc
            qcs = self.tester.alg.solver.constraint_critic(states, actions, uncertainty=distributional_qc)
            if self.tester.alg.solver.constrained_fcn == 'reachability':
                qcs = self.tester.alg.solver._get_qc(qcs)

            lams = self.tester.alg.solver.multiplier(states, qcs)
            assert lams.shape == qcs.shape

            flatten_lams = lams.cpu().numpy()

            NAME2VALUE = dict(zip(['lam'], [flatten_lams]))
            val = NAME2VALUE[metric].reshape(self.x1_grid.shape)

            min_val_list.append(np.min(val))
            max_val_list.append(np.max(val))
            data2plot.append(val)

            min_val = np.min(min_val_list)
            max_val = np.max(max_val_list)
            min_idx = np.argmin(min_val_list)
            max_idx = np.argmax(max_val_list)
            norm = colors.Normalize(vmin=min_val, vmax=max_val)

            for i in range(len(data2plot)):
                sub_ax = axes

                ct = sub_ax.contourf(
                    self.x1_grid, self.x2_grid, data2plot[i],
                    norm=norm,
                    cmap='rainbow',
                )  
                x2_min = -np.sqrt(2 * (self.x1 + 5))
                x2_max = np.sqrt(2 * (5 - self.x1))
                sub_ax.plot(self.x1, x2_min, 'k--')
                sub_ax.plot(self.x1, x2_max, 'k--')

                # sub_ax.set_yticks(np.linspace(0.5, 1.5, 3))
                ct_list.append(ct)
                axes_list.append(sub_ax)

            # cax = add_right_cax(sub_ax, pad=0.01, width=0.02)
            plt.colorbar(ct_list[0], ax=axes_list,
                         shrink=0.8, pad=0.02)

        fig.supxlabel(r'$x_1$')
        fig.supylabel(r'$x_2$')
        plt.savefig(str(LOGS_DIR / self.test_log_dir / (metric + str(self.tester.alg.epochs_completed.item()) + '.png')), dpi=300)

def main():
    # step 1: load config and model
    cfg, test_log_dir, epochs = build_parser()

    # print(epochs)
    for i, epoch in enumerate(epochs[0]):
        if i == 0:
            vizer = Vizer_set(cfg, test_log_dir, epoch, bound=[-6., 6., -6., 6.])
        else:
            vizer.tester.load_model(epoch)
        vizer.plot_region(['lam'])


if __name__ == '__main__':
    # Usage: in the command line, input the followings
    # $ python viz_lam_di.py --set env_name double_integrator --motivation lam --run-dir <log_dir> --epoch <epoch_id, can be more than 1>
    main()