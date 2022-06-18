#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import copy
import logging
import os
from functools import reduce
from copy import deepcopy

import gym
import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

import numpy as np
import random

from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)

def cal_gamma_return_of_an_episode(reward_list, gamma, reward_shift=None, reward_scale=None):
    reward_scale = 1. if reward_scale is None else reward_scale
    reward_shift = 0. if reward_shift is None else reward_shift
    n = len(reward_list)
    gamma_list = np.array([np.power(gamma, i) for i in range(n)])
    reward_list = (np.array(reward_list)+reward_shift)*reward_scale
    gamma_return = np.array([sum(reward_list[i:] * gamma_list[:(n - i)]) for i in range(n)])
    return gamma_return


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args

        if self.args.env_id == 'quadrotor':
            self.env = make('quadrotor', **self.args.config_eval.quadrotor_config)
            self.eval_start_location = self.args.eval_start_location
            assert self.args.num_eval_episode % len(self.eval_start_location) == 0, \
                print('num_epi:', self.args.num_eval_episode, 'len(starting loc):', len(self.eval_start_location))
        else:
            self.env = gym.make(env_id)

        self.policy_with_value = policy_cls(self.env.observation_space, self.env.action_space, self.args)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.args.constrained:
            self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                            self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                            gamma=self.args.gamma,
                                            cost_ptype=self.args.cost_preprocess_type, cost_scale=self.args.cost_scale)
        else:
            self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                            self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                            gamma=self.args.gamma)
        
        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

        self.seed = self.args.seed
        def set_seed(seed):
            self.tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)
        set_seed(self.seed)

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, mode, epi_idx, render=True):
        info_dict = dict()
        eval_v_list = []
        obs_list = []
        action_list = []
        reward_list = []
        cost_list = []
        info_list = []
        done = 0

        if self.args.env_id == 'quadrotor':
            self.env.INIT_X = self.eval_start_location[epi_idx][0]
            self.env.INIT_Z = self.eval_start_location[epi_idx][1]
            self.env.seed(self.args.seed + epi_idx)

        obs = self.env.reset()
        if render: self.env.render()
        while (not done) and len(reward_list) < self.args.max_step:
            processed_obs = self.preprocessor.tf_process_obses(obs[np.newaxis, :])
            if mode == 'Performance':
                action = self.policy_with_value.compute_mode(processed_obs)
            else:
                action, _ = self.policy_with_value.compute_action(processed_obs)
            eval_v = self.policy_with_value.compute_vf(processed_obs)
            eval_v_list.append(eval_v.numpy()[0])
            obs_list.append(obs)
            action_list.append(action.numpy()[0])
            obs, reward, done, info = self.env.step(action.numpy()[0])
            cost = np.max(info.get('constraint_values' if self.args.env_id == 'quadrotor' else 'cost', 0))  # todo: scg: constraint_value; gym: cost
            cost = np.maximum(0., cost)
            if render: self.env.render()
            
            reward_list.append(reward)
            cost_list.append(cost)
            info_list.append(info)
        if mode == 'Performance':
            episode_return = sum(reward_list)
            episode_len = len(reward_list)
            episode_cost_sum = sum(cost_list)
            info_dict = dict(episode_return=episode_return,
                             episode_len=episode_len,
                             episode_cost_sum=episode_cost_sum)
            for key in info_list[0].keys():
                info_key = list(map(lambda x: max(x[key]) if key == 'constraint_values' else x[key], info_list))
                mean_key = sum(info_key) / len(info_key)
                info_dict.update({key: mean_key})
        elif mode == 'Evaluation':
            true_v_list = list(cal_gamma_return_of_an_episode(reward_list, self.args.gamma, self.args.reward_shift,
                                                              self.args.reward_scale))
            info_dict = dict(true_v_list=true_v_list,
                             eval_v_list=eval_v_list)
        return info_dict

    def average_max_n(self, list_for_average, n=None):
        if n is None:
            return sum(list_for_average) / len(list_for_average)
        else:
            sorted_list = sorted(list_for_average, reverse=True)
            return sum(sorted_list[:n]) / n

    def run_n_episodes(self, n, mode):
        epinfo_list = []
        for i in range(n):
            logger.info('logging {}-th episode'.format(i))
            episode_info = self.run_an_episode(mode, i, self.args.eval_render)
            epinfo_list.append(self._metrics_for_an_episode(episode_info) if mode == 'Performance' \
                                                                          else episode_info)
        if mode == 'Performance':
            # n_episode_return_list = [epinfo['episode_return'] for epinfo in epinfo_list]
            # n_episode_len_list = [epinfo['episode_len'] for epinfo in epinfo_list]
            # average_return_with_diff_base = np.array([self.average_max_n(n_episode_return_list, x) for x in [1, 3, 5]])
            # average_len = self.average_max_n(n_episode_len_list)
            # return dict(average_len=average_len,
            #             average_return_with_max1=average_return_with_diff_base[0],
            #             average_return_with_max3=average_return_with_diff_base[1],
            #             average_return_with_max5=average_return_with_diff_base[2],)
            out = {}
            for key in epinfo_list[0].keys():
                value_list = list(map(lambda x: x[key], epinfo_list))
                out.update({key: sum(value_list)/len(value_list)})
            return out
        elif mode == 'Evaluation':
            n_episode_true_v_list = [epinfo['true_v_list'] for epinfo in epinfo_list]
            n_episode_eval_v_list = [epinfo['eval_v_list'] for epinfo in epinfo_list]
            def concat_interest_epi_part_of_one_ite_and_mean(list_of_n_epi, max_state=200):
                tmp = list(copy.deepcopy(list_of_n_epi))
                tmp[0] = tmp[0] if len(tmp[0]) <= max_state else tmp[0][:max_state]

                def reduce_fuc(a, b):
                    return np.concatenate([a, b]) if len(b) < max_state else np.concatenate([a, b[:max_state]])

                interest_epi_part_of_one_ite = reduce(reduce_fuc, tmp)
                return np.mean(interest_epi_part_of_one_ite)
            true_v_mean = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_true_v_list))
            eval_v_mean = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_eval_v_list))
            return dict(true_v_mean=true_v_mean,
                        eval_v_mean=eval_v_mean)

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def _metrics_for_an_episode(self, episode_info):  # user defined, transform episode info dict to metric dict
        episode_return = episode_info['episode_return']
        episode_len = episode_info['episode_len']
        key_list = ['episode_return', 'episode_len', 'avg_reward']
        value_list = [episode_return, episode_len, episode_return / (1e-8+episode_len)]
        if self.args.env_id[:4] == 'Safe':
            episode_cost = episode_info['episode_cost']
            ep_cost_rate = episode_info['ep_cost_rate']
            key_list.extend(['episode_cost', 'ep_cost_rate'])
            value_list.extend([episode_cost, ep_cost_rate])

        elif self.args.env_id == 'quadrotor':
            episode_cost_sum = episode_info['episode_cost_sum']
            episode_mse = episode_info['mse']
            episode_mse_speed = episode_info['mse_speed']
            episode_mse_angle = episode_info['mse_angle']
            episode_mse_angle_speed = episode_info['mse_angle_speed']
            episode_constraint_violation = episode_info['constraint_violation']
            key_list.extend(['episode_cost_sum', 'episode_mse', 'mse_speed', 'mse_angle',
                             'mse_angle_speed', 'episode_constraint_violation'])
            value_list.extend([episode_cost_sum, episode_mse, episode_mse_speed, episode_mse_angle,
                               episode_mse_angle_speed, episode_constraint_violation])

        return dict(zip(key_list, value_list))

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            mean_metric_dict = self.run_n_episodes(self.args.num_eval_episode, 'Performance')
            mean_metric_dict1 = self.run_n_episodes(self.args.num_eval_episode, 'Evaluation')
            mean_metric_dict.update(mean_metric_dict1)
            with self.writer.as_default():
                for key, val in mean_metric_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), mean_metric_dict))
        self.eval_times += 1


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)

def test_evaluator():
    from train_script import built_SAC_parser
    from policy import PolicyWithQs
    args = built_SAC_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.run_evaluation(3)

if __name__ == '__main__':
    test_evaluator()
