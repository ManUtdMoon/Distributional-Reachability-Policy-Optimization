from pathlib import Path

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
LOGS_DIR = ROOT_DIR / 'logs' / 'cartpole-move'


params={'font.family': 'Arial',
        # 'font.serif': 'Times New Roman',
        # 'font.style': 'italic',
        # 'font.weight': 'normal', #or 'blod'
        'font.size': 15,  # or large,small
        }
rcParams.update(params)


class Vizer_set(object):
    def __init__(self,
                 cfg,
                 test_log_dir,
                 epoch,
                 bound=(-0.9, 0.9, -0.2, 0.2)):
        # 1 Load params and models
        tester = Tester(cfg, test_log_dir, epoch)
        self.tester = tester
        self.test_log_dir = test_log_dir

        # 2 Generate batch observations
        x = np.linspace(bound[0], bound[1], 100)
        theta = np.linspace(bound[2], bound[3], 100)
        X, THETA = np.meshgrid(x, theta)
        flatten_x = X.ravel()
        flatten_theta = THETA.ravel()
        batch_obses = np.zeros((len(flatten_x), 4), dtype=np.float32)  # (100*100, 4)
        assert batch_obses.shape == (100*100, 4)
        batch_obses[:, 0] = flatten_x
        batch_obses[:, 1] = flatten_theta

        self.obses = batch_obses
        self.X = X
        self.THETA = THETA
        self.bound = bound
    
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
            assert metric in ['lam']

        fig, axes = plt.subplots(nrows=len(metrics), ncols=1,
                                 figsize=(4, 3),
                                 constrained_layout=True)
        # axes.set_position([0.1, 0.1, 0.9, 0.9])
        # fig_aux = plt.figure()

        for j, metric in enumerate(metrics):
            axes_list = []
            min_val_list = []
            max_val_list = []
            data2plot = []
            ct_list = []

            states = torch.from_numpy(self.obses).float().to(device)
            actions = self.tester.alg.solver.act(states, eval=True)
            assert self.tester.cfg["alg_cfg"]["sac_cfg"]["mlp_multiplier"]
            lams = self.tester.alg.solver.multiplier(states)

            flatten_lams = lams.cpu().numpy()

            NAME2VALUE = dict(zip(['lam'], [flatten_lams]))
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
                sub_ax = axes

                ct = sub_ax.contourf(self.X, self.THETA, data2plot[i],
                                     norm=norm,
                                     cmap='rainbow',
                                    #  levels=[-0.06, -0.04, -0.02, 0., 0.2, 0.6, 1.0, 1.2],  # CBF
                                    #  levels=[-2.4, -1.2, 0., 1.2, 2.4, 3.6, 4.8],  # RAC
                                    #  levels=[-1.2, -0.8, -0.4, 0., 0.4, 0.8, 1.2],  # SI
                                     )
                ct_list.append(ct)
                axes_list.append(sub_ax)

                # plot constrained boundary
                verts = [
                    (-0.9, -0.2),  # left, bottom
                    (0.9, -0.2),  # left, top
                    (0.9, 0.2),  # right, top
                    (-0.9, 0.2),  # right, bottom
                    (-0.9, -0.2),  # ignored
                ]

                codes = [
                    matplotlib.path.Path.MOVETO,
                    matplotlib.path.Path.LINETO,
                    matplotlib.path.Path.LINETO,
                    matplotlib.path.Path.LINETO,
                    matplotlib.path.Path.CLOSEPOLY,
                ]

                path = matplotlib.path.Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=(1., 1., 1., 0.1), lw=1)
                sub_ax.add_patch(patch)

            # cax = add_right_cax(sub_ax, pad=0.01, width=0.02)
            plt.colorbar(ct_list[0], ax=axes_list,
                         shrink=0.8, pad=0.02)

        fig.supxlabel('x')
        fig.supylabel(r'$\theta$')
        plt.savefig(str(LOGS_DIR / self.test_log_dir / (metric + str(self.tester.alg.epochs_completed.item()) + '.png')), dpi=300)

def main():
    # step 1: load config and model
    cfg, test_log_dir, epochs = build_parser()

    # print(epochs)
    for i, epoch in enumerate(epochs[0]):
        if i == 0:
            vizer = Vizer_set(cfg, test_log_dir, epoch, bound=[-1., 1., -0.3, 0.3])
        else:
            vizer.tester.load_model(epoch)
        vizer.plot_region(['lam'])


if __name__ == '__main__':
    # Usage: in the command line, input the followings
    # $ export PYTHONPATH=$PYTHONPATH:/your/path/to/Safe_MBRL
    # $ python viz_multiplier_cartpole.py --run-dir <log_dir> --set env_name cartpole-move --epoch <epoch_id, can be more than 1>
    main()