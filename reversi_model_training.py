"""
Collection of functions for training a single model using some RL algorithm
"""

import argparse
from baselines_reversi import deepq
from baselines.common.tf_util import save_variables, load_variables

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def smooth(y, box_pts):
    """
    For smoothing the graph.
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def win_rate(rew):
    rew = np.array([rew])
    wins = np.cumsum(rew >= 1)
    total = np.cumsum(np.ones(rew.shape[1]))
    return np.divide(wins, total)


def dqn_reversi(env, network, config, **kwargs):
    """
    Parameters:
    env:      gym environment
    network:  Q function approximator. Takes an observation tensor and return
              Q values. Can also be a string, which will use a predefined
              network. Available networks = ['mlp', 'cnn', 'cnn_small', 'lstm',
              'cnn_lstm', 'cnn_lnlstm', 'conv_only'].
    """
    def callback(locals, globals):
        # Training stops when this function returns True
        # TODO: Add stopping condition

        # print(locals['obs'][:,:,1] - locals['obs'][:,:,2])
        # print(locals['debug']['q_values'](locals['obs']).reshape((8,8)))
        # print(tf.random.multinomial(locals['obs'][:,:,3].reshape((1, 64)), 1))

        if config['load_model'] and locals['t'] == 0:
            load_variables(config['load_path'])
        if locals['t'] == config['total_timesteps'] - 1:
            if config['vis']:
                rew = locals['episode_rewards']
                smooth_rew = smooth(rew, 500)
                plt.plot(range(len(smooth_rew)), smooth_rew)
                plt.show()
                plt.plot(win_rate(rew))
                plt.show()
            return True
        if config['save_model'] and locals['t'] % config['save_timesteps'] == 0:
            save_variables(config['save_path'])
        if locals['t'] % config['opponent_update'] == 0:
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope="deepq/q_func")
            env.opponent_model.update(q_func_vars)
        return False

    act = deepq.learn(
        env,
        network=network,
        lr=config['lr'],
        total_timesteps=config['total_timesteps'],
        buffer_size=config['buffer_size'],
        exploration_fraction=config['exploration_fraction'],
        exploration_final_eps=config['exploration_final_eps'],
        print_freq=config['print_freq'],
        callback=callback,
        **kwargs
    )
