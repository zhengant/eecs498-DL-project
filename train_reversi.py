import gym
import tensorflow as tf
from models import reversi_network

from baselines.common.tf_util import load_variables

from reversi_environment import ReversiEnvironment
from reversi_agents import RandomAgent, ClonedAgent, PlayerAgent
from reversi_model_training import dqn_reversi

model_loc = 'saved_models/reversi_1.pkl'

if __name__ == '__main__':
    num_layers = 3
    num_hidden = 128
    activation = tf.nn.relu
    layer_norm = False

    config = {
        'lr': 1e-2,
        'total_timesteps': 100000,
        'buffer_size': 50000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.02,
        'print_freq': 1,
        'vis': True,

        'save_model': False,
        'save_path': model_loc,
        'save_timesteps': 10,

        'load_model': False,
        'load_path': model_loc,

        'opponent_update': 1000,
    }
    kwargs = {
        'num_layers': num_layers,
        'num_hidden': num_hidden,
        'activation': activation,
        # 'activation': tf.tanh,
        'layer_norm': layer_norm,
        'gamma': 1
    }
    env = ReversiEnvironment()
    # env.update_opponent_model(ClonedAgent(reversi_network(num_layers=num_layers,
    #                                                       num_hidden=num_hidden,
    #                                                       activation=activation,
    #                                                       layer_norm=layer_norm),
    #                                       env))
    # env.update_opponent_model(PlayerAgent())
    env.update_opponent_model(RandomAgent())
    dqn_reversi(env, 'reversi_network', config, **kwargs)
