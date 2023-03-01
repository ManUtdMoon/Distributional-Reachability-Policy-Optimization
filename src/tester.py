from argparse import ArgumentParser
import os
import json
import datetime
import time
from pathlib import Path
import sys

PROJ_DIR = Path.cwd().parent
sys.path.append(str(PROJ_DIR))

import numpy as np
from pyrsistent import v

import torch

from src import cli
from src.defaults import ROOT_DIR
from src.log import default_log as log, TabularLog
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.torch_util import device
from src.shared import get_env
from src.smbpo import SMBPO
from src.sampling import BaseBatchedEnv, ProductEnv, env_dims, isdiscrete, SafetySampleBuffer


ROOT_DIR = Path(ROOT_DIR)
assert ROOT_DIR.is_dir(), ROOT_DIR

@torch.no_grad()
def sample_episodes_batched_with_infos(env, policy, n_traj, eval=False, safe_shield_threshold=-0.1, shield_type="none", print_lam=False):
    if not isinstance(env, BaseBatchedEnv):
        env = ProductEnv([env])

    state_dim, action_dim, _ = env_dims(env)
    discrete_actions = isdiscrete(env.action_space)
    traj_buffer_factory = lambda: SafetySampleBuffer(state_dim, 1 if discrete_actions else action_dim, env._max_episode_steps,
                                                     discrete_actions=discrete_actions)
    traj_buffers = [traj_buffer_factory() for _ in range(env.n_envs)]
    info_buffers = [[] for _ in range(env.n_envs)]
    complete_episodes = []
    complete_info_per_episode = []

    states = env.reset()
    while True:
        # compute the time consumed to get the action
        t0 = time.time()
        actions_performance = policy.act(states, eval=eval)
        # -------- safety shield ----------- #
        if eval:
            qcs = policy._get_qc(policy.constraint_critic(states, actions_performance))
            actions_safe = policy.actor_safe.act(states, eval=eval)
            if print_lam:
                safe_qcs = policy._get_qc(policy.constraint_critic(states, actions_safe))
                lams = policy.multiplier(states, safe_qcs)
                print(lams.cpu().numpy().squeeze())
            if shield_type == "safe":
                danger_bool = (qcs > safe_shield_threshold).tile((action_dim, 1)).t()
                actions = torch.where(danger_bool, actions_safe, actions_performance)
            elif shield_type == "linear":
                actions = actions_safe
                for i in range(11):
                    ratio = (10-i)/10  # 1.0~0.0
                    action_mix = actions_safe*ratio + actions_performance*(1-ratio)
                    qcs_mix = policy._get_qc(policy.constraint_critic(states, action_mix))
                    safe_bool = (qcs_mix <= safe_shield_threshold).tile((action_dim, 1)).t()
                    actions = torch.where(safe_bool, action_mix, actions)
            else:
                actions = actions_performance
        t1 = time.time()

        next_states, rewards, dones, infos = env.step(actions)
        violations = [info['violation'] for info in infos]

        _next_states = next_states.clone()
        reset_indices = []

        for i in range(env.n_envs):
            if i == 0:
                infos[i].update({'time': t1-t0})
            traj_buffers[i].append(states=states[i], actions=actions[i], next_states=next_states[i],
                                   rewards=rewards[i], dones=dones[i], violations=violations[i])
            info_buffers[i].append(infos[i])
            if dones[i] or len(traj_buffers[i]) == env._max_episode_steps:
                complete_episodes.append(traj_buffers[i])
                complete_info_per_episode.append(
                    dict(
                        zip(infos[i].keys(), [[info[key] for info in info_buffers[i]] for key in infos[i].keys()])
                    )
                )
                if len(complete_episodes) == n_traj:
                    # Done!
                    assert len(complete_info_per_episode) == n_traj
                    return complete_episodes, complete_info_per_episode

                reset_indices.append(i)
                traj_buffers[i] = traj_buffer_factory()
                info_buffers[i] = []

        if reset_indices:
            reset_indices = np.array(reset_indices)
            _next_states[reset_indices] = env.partial_reset(reset_indices)

        states.copy_(_next_states)

def mpc_sample_episodes_batched_with_infos(env, controller, n_traj, eval=True):

    state_dim, action_dim, _ = env_dims(env)
    discrete_actions = isdiscrete(env.action_space)
    traj_buffer_factory = lambda: SafetySampleBuffer(state_dim, 1 if discrete_actions else action_dim, env.max_episode_steps,
                                                     discrete_actions=discrete_actions)
    traj_buffers = [traj_buffer_factory() for _ in range(1)]
    info_buffers = [[] for _ in range(1)]
    complete_episodes = []
    complete_info_per_episode = []
    states, infos = env.reset(return_info = True)
    while True:
        # compute the time when the controller get the actions
        t0 = time.time()
        actions = controller(states, infos)
        t1 = time.time()

        next_states, rewards, dones, infos = env.step(actions)
        infos.update({'time': t1-t0})
        # violations = [info['violation'] for info in infos]
        violations = [False]
        _next_states = next_states.copy()
        reset_indices = []

        for i in range(1):
            traj_buffers[i].append(states=torch.tensor(states), actions=torch.tensor(actions), next_states=torch.tensor(next_states),
                                   rewards=torch.tensor(rewards), dones=torch.tensor(dones), violations=torch.tensor(violations))
            info_buffers[i].append(infos)
            if dones or len(traj_buffers[i]) == env._max_episode_steps:
                complete_episodes.append(traj_buffers[i])
                complete_info_per_episode.append(
                    dict(
                        zip(infos.keys(), [[info[key] for info in info_buffers[i]] for key in infos.keys()])
                    )
                )
                if len(complete_episodes) == n_traj:
                    # Done!
                    assert len(complete_info_per_episode) == n_traj
                    return complete_episodes, complete_info_per_episode

                reset_indices.append(i)
                traj_buffers[i] = traj_buffer_factory()
                info_buffers[i] = []

        if reset_indices:
            reset_indices = np.array(reset_indices)
            _next_states[reset_indices] = env.partial_reset(reset_indices)

        states = _next_states.copy()

class Config(BaseConfig):
    env_name = Require(str)
    env_cfg = {}
    seed = 1
    epochs = 600
    alg_cfg = SMBPO.Config()


def build_parser():
    # load saved config
    cfg = Config()

    parser = ArgumentParser()
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--set', default=[], action='append', nargs=2)
    parser.add_argument('--epoch', default=[], nargs='*', type=int, action='append')
    cli_args = parser.parse_args()
    set_args = dict(cli_args.set)

    # Directory structure: ROOT_DIR / logs / env_name / run-dir
    root_log_dir = ROOT_DIR / 'logs'
    
    assert 'env_name' in set_args, 'Must specify env_name if using --resume'
    assert cli_args.run_dir is not None, 'Must specify --run-dir if using --resume'
    run_dir = root_log_dir / set_args['env_name'] / cli_args.run_dir
    assert run_dir.is_dir(), f'Run directory does not exist: {run_dir}'

    with (run_dir / 'config.json').open('r') as f:
        saved_cfg = json.load(f)
        assert set_args['env_name'] == saved_cfg["env_name"]
        cfg.update(saved_cfg)

    cfg.alg_cfg.update(dict(mode='test'))

    # Ensure all required arguments have been set
    cfg.verify()
    for attr in ('env_name', 'seed'):
        assert hasattr(cfg, attr), f'Config must specify {attr}'
    
    # main body    
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = run_dir / 'test-{}'.format(time_now)
    test_log_dir.mkdir(exist_ok=True, parents=True)

    return cfg, test_log_dir, cli_args.epoch


class Tester(object):
    def __init__(self, cfg, test_log_dir, epoch_id):
        env_factory = lambda id=None: get_env(cfg.env_name, **{**cfg.env_cfg, **dict(id=id)})
    
        log.setup(test_log_dir)
        log.message(f'Test log directory: {test_log_dir}')
        self.cfg = cfg
        self.test_log_dir = test_log_dir

        self.data = CheckpointableData()
        self.alg = SMBPO(cfg.alg_cfg, env_factory, self.data, cfg.epochs)
        self.checkpointer = Checkpointer(self.alg, log.dir.parent, 'ckpt_{}.pt')
        self.data_checkpointer = Checkpointer(self.data, log.dir.parent, 'data.pt')
        self.load_model(epoch_id)

    def load_model(self, epoch_id):
        self.epoch_id = epoch_id
        # Check if existing run
        if self.data_checkpointer.try_load():
            log('Data load succeeded')
            loaded_epoch = self.checkpointer.load_latest([epoch_id])
            if isinstance(loaded_epoch, int):
                assert loaded_epoch == self.alg.epochs_completed
                log(f'Solver load epoch {epoch_id} succeeded')
            else:
                assert self.alg.epochs_completed == 0
                log('Solver load failed')
        else:
            log('Data load failed')

    def run_evaluation(self, print_lam=False):
        test_traj, info_per_traj = sample_episodes_batched_with_infos(
            self.alg.eval_env,
            self.alg.solver, 1, eval=True,
            print_lam=print_lam,
            shield_type='linear'
        )

        lengths = [len(traj) for traj in test_traj]
        length_mean = float(np.mean(lengths))

        returns = [traj.get('rewards').sum().item() for traj in test_traj]
        return_mean = float(np.mean(returns))

        log.message(f'test length mean: {length_mean}')
        log.message(f'test return mean: {return_mean}')
        
        if self.cfg.env_name == 'quadrotor':
            for key in info_per_traj[0].keys():
                tmp_list = info_per_traj[0][key]
                if (not isinstance(tmp_list[0], list)) and (not isinstance(tmp_list[0], np.ndarray)):
                    log.message(f'avg_{key}: {sum(tmp_list)/len(tmp_list)}')
        
        return test_traj, info_per_traj

    def post_process(self, trajs):
        '''env-specific post process, incl. saving data
        '''
        if self.cfg.env_name == 'quadrotor':  # only need one traj
            states = trajs[0].get('states').cpu().numpy()
            x = states[:, 0]
            z = states[:, 2]

            np.save(self.test_log_dir / 'coordinates_x_z.npy', np.array([dict(x=x, z=z)]))

        elif self.cfg.env_name == 'cartpole-move':
            states = trajs[0].get('states').cpu().numpy()
            x = states[:, 0]
            theta = states[:, 1]

            np.save(self.test_log_dir / 'traj_' + str(self.epoch_id) + '.npy', np.array([dict(x=x, theta=theta)]))

def main():
    # step 1: load config and model
    cfg, test_log_dir, epochs = build_parser()
    for epoch in epochs[0]:
        tester = Tester(cfg, test_log_dir, epoch)

        # step 2: run evaluation
        test_traj, _ = tester.run_evaluation()

        # step 3: post-evaluating, record, save, print
        tester.post_process(test_traj)
        break # only the first epoch are recorded


if __name__ == '__main__':
    # Usage: in the command line, input the followings
    # $ python tester.py --run-dir 06-03-22_15.45.15_mbpo --set env_name quadrotor --epoch <epoch_id>
    main()