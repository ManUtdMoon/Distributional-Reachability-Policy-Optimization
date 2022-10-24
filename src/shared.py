import numpy as np
import torch
from gym.wrappers import RescaleAction


def get_env(env_name, wrap_torch=True, **kwargs):
    from .env.torch_wrapper import TorchWrapper
    from .env.double_integrator import DoubleIntegrator
    envs = {
        'double_integrator': DoubleIntegrator,
    }
    # if env_name != 'quadrotor':
    #     assert 'id' in kwargs.keys()
    #     kwargs.pop('id')  # the keyword arg 'mode' is only valid for quadrotor env

    env = envs[env_name](**kwargs)
    if not (np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)):
        env = RescaleAction(env, -1.0, 1.0)
    if wrap_torch:
        env = TorchWrapper(env)
    return env
