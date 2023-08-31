import numpy as np
import torch
from gym.wrappers import RescaleAction


def get_env(env_name, wrap_torch=True, rescale_action = True, **kwargs):
    from .env.torch_wrapper import TorchWrapper
    from .env.hopper_no_bonus import HopperNoBonusEnv
    from .env.cheetah_no_flip import CheetahNoFlipEnv
    from .env.ant_no_bonus import AntNoBonusEnv
    from .env.humanoid_no_bonus import HumanoidNoBonusEnv
    from .env.poles.classic_pendulum import SafeClassicPendulum
    from .env.poles.inverted_pendulum import SafeInvertedPendulumEnv
    from .env.quadrotor.quadrotor import QuadrotorWrapperEnv
    from .env.tracking.pyth_veh3dofconti_surrcstr_data import SimuVeh3dofcontiSurrCstr
    from .env.tracking.pyth_veh3dofconti_surrcstr_data4mpc import SimuVeh3dofcontiSurrCstr2
    from .env.point_robot import PointRobot
    envs = {
        'hopper': HopperNoBonusEnv,
        'cheetah-no-flip': CheetahNoFlipEnv,
        'ant': AntNoBonusEnv,
        'humanoid': HumanoidNoBonusEnv,
        'pendulum-upright': SafeClassicPendulum,
        'pendulum-tilt': SafeClassicPendulum,
        'cartpole-upright': SafeInvertedPendulumEnv,
        'cartpole-move': SafeInvertedPendulumEnv,
        'quadrotor': QuadrotorWrapperEnv,
        'tracking': SimuVeh3dofcontiSurrCstr,
        'tracking_model': SimuVeh3dofcontiSurrCstr2,
        'point-robot': PointRobot,
    }
    # if env_name != 'quadrotor':
    #     assert 'id' in kwargs.keys()
    #     kwargs.pop('id')  # the keyword arg 'mode' is only valid for quadrotor env

    env = envs[env_name](**kwargs)
    if not (np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)) and rescale_action:
        env = RescaleAction(env, -1.0, 1.0)
    if wrap_torch:
        env = TorchWrapper(env)
    return env
