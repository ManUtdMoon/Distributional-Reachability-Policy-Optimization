from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


from src.defaults import ROOT_DIR


ROOT_DIR = Path(ROOT_DIR)
assert ROOT_DIR.is_dir(), ROOT_DIR
LOGS_DIR = ROOT_DIR / 'logs' / 'quadrotor'


ALG2CMAP = dict([('RAC (ours)', (0.0, 0.24705882352941178, 1.0)),
                 ('MBPO-Lagrangian', (0.011764705882352941, 0.9294117647058824, 0.22745098039215686)),
                 ('SAC-Reward Shaping', (0.9098039215686274, 0.0, 0.043137254901960784)),
                 ('SAC-CBF', (0.5411764705882353, 0.16862745098039217, 0.8862745098039215)),
                 ('SAC-SI', (1.0, 0.7686274509803922, 0.0))])

params={'font.family': 'Arial',
        # 'font.serif': 'Times New Roman',
        # 'font.style': 'italic',
        # 'font.weight': 'normal', #or 'blod'
        'font.size': 15,  # or large,small
        }
rcParams.update(params)


def plt_trajectory(ax, alg, trj_dir):
    coordinates = np.load(trj_dir / 'coordinates_x_z.npy', allow_pickle=True)[0]

    xs = coordinates['x']
    zs = coordinates['z']

    
    print(xs[0], zs[0])
    trj = ax.plot(xs, zs, c=ALG2CMAP[alg],
                    linewidth=2,
                    label=alg,)
    # plt.colorbar(trj)
    # plt.colorbar(PID_trj_plot)
    # plt.colorbar(PID_ref_plot)

    return trj


def plt_ref_trj(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1.
    offset_x = 0.
    offset_z = 1.
    x = radius * np.cos(theta) + offset_x
    z = radius * np.sin(theta) + offset_z

    ref = ax.plot(x, z, c='Black', label='Reference', linewidth=2.,
                  ls='--', markersize=0.1)

    return ref


def plt_constraint(ax):
    x = np.linspace(-4, 4, 100)
    z_lb = 0.5 * np.ones_like(x)
    z_ub = 1.5 * np.ones_like(x)

    ax.plot(x, z_lb, ls='solid', c='Black')
    ax.plot(x, z_ub, ls='solid', c='Black', label='Constraints')


if __name__ == '__main__':
    fig = plt.figure()  #figsize=[6, 6]
    ax = plt.axes([0.2, 0.2, 0.7, 0.7])

    # MBPO-L
    trj_MBPO = plt_trajectory(
        ax, 'MBPO-Lagrangian',
        LOGS_DIR / '06-25-22_23.40.00_eejl' / 'test-2022-06-26-08-08-56'
    )

    # Plot constraint and ref
    ref = plt_ref_trj(ax)
    plt_constraint(ax)

    # Plot settings
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 2.2)

    # legend1 = ax.legend(*trj_RAC.legend_elements(),
    #                     loc="lower left", title="Algorithms")
    ax.legend(frameon=False,
              fontsize=14,
              bbox_to_anchor=(0.5, -0.25),
              loc='lower center', ncol=5)
    
    plt.title('Quadrotor Tracking Trajectories Visualization', fontsize=14)
    # plt.tight_layout(pad=0.5)
    # plt.show()
    plt.savefig("./traj.png", dpi=300)