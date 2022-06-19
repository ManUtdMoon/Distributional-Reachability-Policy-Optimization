from argparse import ArgumentParser
import os
import json
import datetime
from pathlib import Path

import numpy as np

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


def sample_episodes_batched_with_infos(env, policy, n_traj, eval=False):
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
        actions = policy.act(states, eval=eval)
        next_states, rewards, dones, infos = env.step(actions)
        violations = [info['violation'] for info in infos]

        _next_states = next_states.clone()
        reset_indices = []

        for i in range(env.n_envs):
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

    return cfg, run_dir, test_log_dir


class Tester(object):
    def __init__(self, cfg, test_log_dir, epoch_id):
        env_factory = lambda id=None: get_env(cfg.env_name, **{**cfg.env_cfg, **dict(id=id)})
    
        log.setup(test_log_dir)
        log.message(f'Test log directory: {test_log_dir}')
        self.cfg = cfg
        self.test_log_dir = test_log_dir

        self.data = CheckpointableData()
        self.alg = SMBPO(cfg.alg_cfg, env_factory, self.data)
        self.checkpointer = Checkpointer(self.alg, log.dir.parent, 'ckpt_{}.pt')
        self.data_checkpointer = Checkpointer(self.data, log.dir.parent, 'data.pt')

        # Check if existing run
        if self.data_checkpointer.try_load():
            log('Data load succeeded')
            loaded_epoch = self.checkpointer.load_latest([epoch_id])
            if isinstance(loaded_epoch, int):
                assert loaded_epoch == self.alg.epochs_completed
                log('Solver load succeeded')
            else:
                assert self.alg.epochs_completed == 0
                log('Solver load failed')
        else:
            log('Data load failed')

    def run_evaluation(self):
        test_traj, info_per_traj = sample_episodes_batched_with_infos(
            self.alg.eval_env,
            self.alg.solver, 1, eval=True
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


def main(epoch_id):
    # step 1: load config and model
    cfg, rundir, test_log_dir = build_parser()
    tester = Tester(cfg, test_log_dir, epoch_id)

    # step 2: run evaluation
    test_traj, _ = tester.run_evaluation()

    # step 3: post-evaluating, record, save, print
    tester.post_process(test_traj)


if __name__ == '__main__':
    main(epoch_id=60)