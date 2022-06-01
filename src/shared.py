import numpy as np
import torch
from gym.wrappers import RescaleAction

from .sampling import SampleBuffer


def get_env(env_name, wrap_torch=True, **kwargs):
    from .env.torch_wrapper import TorchWrapper
    from .env.hopper_no_bonus import HopperNoBonusEnv
    from .env.cheetah_no_flip import CheetahNoFlipEnv
    from .env.ant_no_bonus import AntNoBonusEnv
    from .env.humanoid_no_bonus import HumanoidNoBonusEnv
    from .env.poles.classic_pendulum import SafeClassicPendulum
    envs = {
        'hopper': HopperNoBonusEnv,
        'cheetah-no-flip': CheetahNoFlipEnv,
        'ant': AntNoBonusEnv,
        'humanoid': HumanoidNoBonusEnv,
        'pendulum-upright': SafeClassicPendulum,
    }
    env = envs[env_name](**kwargs)
    if not (np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)):
        env = RescaleAction(env, -1.0, 1.0)
    if wrap_torch:
        env = TorchWrapper(env)
    return env


class SafetySampleBuffer(SampleBuffer):
    COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, 'violations')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_buffer('violations', torch.bool, [])