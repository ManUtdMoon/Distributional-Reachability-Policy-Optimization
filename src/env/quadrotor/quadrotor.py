import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from gym import Wrapper

import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from src.env.poles.constraints import BoundedConstraint, ConstrainedVariableType
from src.defaults import ROOT_DIR

ROOT_DIR = Path(ROOT_DIR)
ENV_DIR = ROOT_DIR / 'src' / 'env' / 'quadrotor'
train_cfg_yaml = ENV_DIR / 'constrained_tracking_reset.yaml'
eval_cfg_yaml = ENV_DIR / 'constrained_tracking_eval.yaml'

assert train_cfg_yaml.is_file() and eval_cfg_yaml.is_file()

CONFIG_FACTORY = ConfigFactory()
CONFIG_FACTORY.parser.set_defaults(overrides=[str(train_cfg_yaml)])
config = CONFIG_FACTORY.merge()

CONFIG_FACTORY_EVAL = ConfigFactory()
CONFIG_FACTORY_EVAL.parser.set_defaults(overrides=[str(eval_cfg_yaml)])
config_eval = CONFIG_FACTORY_EVAL.merge()

MAX_EPISODE_STEPS = int(config.quadrotor_config['episode_len_sec'] * \
                        config.quadrotor_config['ctrl_freq'])


class QuadrotorWrapperEnv(Wrapper):
    _eval_start_location = [(1., 1.), (-1., 1.), (0., 0.53), (0., 1.47)]

    def __init__(self, id=None) -> None:
        env = make('quadrotor', **config.quadrotor_config) if id is None \
            else make('quadrotor', **config_eval.quadrotor_config)
        super().__init__(env)
        self._max_episode_steps = MAX_EPISODE_STEPS
        self._id = id
        self.env.seed(np.random.randint(2**10))

        cons_cfg = config.quadrotor_config.constraints[0]

        self.constraints = BoundedConstraint(
            self.env.observation_space.shape[0],
            lower_bounds=cons_cfg.lower_bounds,
            upper_bounds=cons_cfg.upper_bounds,
            constrained_variable=ConstrainedVariableType.STATE,
            active_dims=cons_cfg.active_dims
        )

        con_dim = [cons_cfg.active_dims] if isinstance(cons_cfg.active_dims, int) \
            else cons_cfg.active_dims
        self.con_dim = len(con_dim) * 2  # lb & ub
        self.info = None
    
    def step(self, action: np.array):
        new_info = dict(
            constraint_value=self.info['constraint_values'],
        )
        assert np.all(np.allclose(new_info['constraint_value'], self.constraints.get_value(self.state), atol=1e-6)), \
            print(new_info['constraint_value'], self.constraints.get_value(self.state))
        next_obs, rew, done, info = super().step(action)
        new_info.update(dict(
            violation=bool(info['constraint_violation']),
            **info
        ))
        self.state = next_obs
        self.info = new_info  # including: h(s) not h(s')
        return next_obs, rew, done, new_info
    
    def reset(self):
        state, info = super().reset()
        self.state = state
        self.info = info
        if self._id:
            print(self.env._get_observation())
        # if self._id:
        #     self.env.INIT_X = self._eval_start_location[self._id % 4][0]
        #     self.env.INIT_Z = self._eval_start_location[self._id % 4][1]
        return state
    
    def check_done(self, states: np.array):
        '''Compute whether the states are out of bound

        params:
            states shape: (*, dim_s) where * can be any number of dimensions incl. none
        return:
            dones: shape (*,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2
        batch_size = states.shape[:-1]

        x_threshold = self.env.x_threshold
        z_threshold = self.env.z_threshold
        theta_threshold_radians = 85 * math.pi / 180

        x = states[..., 0]
        z = states[..., 2]
        theta = states[..., 4]

        done = (x < -x_threshold) +\
               (x > x_threshold) +\
               (z < -z_threshold) +\
               (z > z_threshold) +\
               (theta < -theta_threshold_radians) +\
               (theta > theta_threshold_radians)
        
        assert done.shape == batch_size
        return done

    def check_violation(self, states: np.array):
        '''Compute whether the constraints are violated

        params:
            states shape: (*, dim_s)
        return:
            violations: shape (*,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        z_ub = 1.5
        z_lb = 0.5

        z = states[..., 2]
        
        # method 1: compute directly
        violations = np.logical_or(
            z < z_lb,
            z > z_ub
        )

        # method 2: use boundedconstraints
        violations_ = self.constraints.is_violated(states)

        assert np.all(violations == violations_)

        return violations

    def get_constraint_values(self, states):
        '''Compute the constraints values given states

        params:
            states shape: (n, dim_s) or (dim_s)
        return:
            constraint_value: shape (n, self.con_dim) or (self.con_dim,)
        '''
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return np.squeeze(self.constraints.get_value(states))


if __name__ == '__main__':
    env = QuadrotorWrapperEnv(
        id=2,
    )
    np.set_printoptions(precision=5, suppress=True)
    s = env.reset()
    print(s)
    for i in range(100):
        act = env.action_space.sample()
        s_p, r, done, info = env.step(act)
        print(act, s_p, r, done, info)
        print(env.check_done(np.tile(s_p, [3, 2, 1])))
        print(env.check_violation(np.tile(s_p, [3, 2, 1])))
        print(env.get_constraint_values(np.tile(s_p, [3, 2, 1])))
        print(f"--------------step {i}---------------")

        if done:
            break