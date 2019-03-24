"""
Collection of functions for training a single model using some RL algorithm
"""

import argparse
from baselines import deepq

import matplotlib.pyplot as plt
import numpy as np

def smooth(y, box_pts):
    """
    For smoothing the graph.
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def dqn(env, network, config, **kwargs):
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
        if locals['t'] == config['total_timesteps'] - 1:
            if config['vis']:
                rew = locals['episode_rewards']
                smooth_rew = smooth(rew, 500)
                plt.plot(range(len(smooth_rew)), smooth_rew)
                plt.show()
            return True
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


def dynaq(env, env_model, action_model):
    # maybe???
    pass


def test_model(env, model):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
        default=1e-3, type=float,
        help='Learning rate')
    parser.add_argument('--total_timesteps',
        default=100000, type=int,
        help='Number of env steps to optimize for')
    parser.add_argument('--buffer_size',
        default=50000, type=int,
        help='Size of replay buffer')
    parser.add_argument('--exploration_fraction',
        default=0.1, type=float,
        help='Fraction of entire training period over which the exploration' + \
             ' rate is annealed')
    parser.add_argument('--exploration_final_eps',
        default=0.02, type=float,
        help='Final value of exploration rate')
    parser.add_argument('--print_freq',
        default=10, type=int,
        help='How often to print out training progress')
    args = parser.parse_args()

    dqn(None, None, vars(args), {}) # TODO: fill in env and network
