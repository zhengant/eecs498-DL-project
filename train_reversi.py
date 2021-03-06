import gym
import tensorflow as tf
from models import reversi_network

import baselines.common.tf_util as U
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import load_variables

from reversi_environment import ReversiEnvironment
from reversi_agents import RandomAgent, ClonedAgent, PlayerAgent, CompositeAgent, ModelLoader
from reversi_model_training import dqn_reversi
from reversi_rewards import simple_rewards, corner_rewards, greedy_rewards, opp_greedy_rewards, opp_mobility_rewards, edge_rewards, mobility_rewards, opp_corner_rewards, opp_edge_rewards

import numpy as np


if __name__ == '__main__':
    num_layers = 3
    num_hidden = 128
    activation = tf.nn.relu
    layer_norm = False
    reward_fn = simple_rewards

    model_name = 'reversi_baseline_4'
    model_loc = "".join(['saved_models/', model_name, '.pkl'])


    env = ReversiEnvironment(reward_fn=reward_fn, reward_before_opp=True)

    # Sub Expert Model
    other_models = None
    env.update_opponent_model(ClonedAgent(reversi_network(num_layers=num_layers,
                                                          num_hidden=num_hidden,
                                                          activation=activation,
                                                          layer_norm=layer_norm),
                                          env,
                                          other_models=[],
                                          model_scope=model_name))

    # Aggregate Model
    # other_models = ModelLoader(env, model_scope=model_name, other_model_names=other_model_names)

    # other_model_names = ['reversi_edges_2',
    #                      'reversi_corners_2',
    #                      'reversi_greedy_2',
    #                      'reversi_mobility_2',
    #                      'reversi_opp_edges_2',
    #                      'reversi_opp_corners_2',
    #                      'reversi_opp_greedy_2',
    #                      'reversi_opp_mobility_2']
                         
    # env.observation_space = gym.spaces.Box(low=0, high=1,
    #                                         shape=(8, 8, 4+len(other_model_names)),
    #                                         dtype=np.int32)
    # env.update_opponent_model(CompositeAgent(reversi_network(num_layers=num_layers,
    #                                                       num_hidden=num_hidden,
    #                                                       activation=activation,
    #                                                       layer_norm=layer_norm),
    #                                       env,
    #                                       other_models=other_models,
    #                                       model_scope=model_name))

    config = {
        'lr': 1e-2,
        'total_timesteps': 350000,
        'buffer_size': 50000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.02,
        'print_freq': 50,
        'vis': True,

        'model_name': model_name,

        'save_model': True,
        'save_path': model_loc,
        'save_timesteps': 100,

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
        'gamma': 1,
        'other_models': other_models,
    }

    dqn_reversi(env, 'reversi_network', config, **kwargs)
