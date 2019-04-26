from reversi_agents import PredictorAgent, RandomAgent, CompositeAgent, ModelLoader, PlayerAgent
from reversi_environment import ReversiEnvironment, legal_moves, random_move
from reversi_rewards import zero_rewards
from models import reversi_network

from baselines.common.tf_util import save_variables, load_variables, get_session

import tensorflow as tf
import numpy as np


def load_model(model_name, env):
    model_load_path = "".join(['saved_models/', model_name, '.pkl'])
    model = PredictorAgent(reversi_network(num_layers=num_layers,
                                          num_hidden=num_hidden,
                                          activation=activation,
                                          layer_norm=layer_norm),
                        env,
                        model_scope=model_name)
    load_variables(model_load_path, variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
    return model


def load_model_2(model_name, env, other_model_names):
    other_models = ModelLoader(env, model_scope=model_name, other_model_names=other_model_names)

    model_load_path = "".join(['saved_models/', model_name, '.pkl'])
    model = CompositeAgent(reversi_network(num_layers=num_layers,
                                          num_hidden=num_hidden,
                                          activation=activation,
                                          layer_norm=layer_norm),
                        env,
                        other_models=other_models.other_models,
                        model_scope=model_name)
    load_variables(model_load_path, variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
    return model



other_model_names = ['reversi_edges', 'reversi_corners', 'reversi_greedy', 'reversi_mobility', 'reversi_opp_greedy', 'reversi_opp_mobility']


num_games = 1000

num_layers = 3
num_hidden = 128
activation = tf.nn.relu
layer_norm = False

sess = get_session()

# model1_name = 'reversi_edges_and_corners'
model1_name = 'reversi_baseline_2'
model2_name = 'reversi_edges_2'
env = ReversiEnvironment(reward_fn=zero_rewards)
# model2 = load_model(model2_name, env)
# model1 = load_model(model1_name, env)

model_load_path = "".join(['saved_models/', 'reversi_edges_2', '.pkl'])
model1 = PredictorAgent(reversi_network(num_layers=num_layers,
                                       num_hidden=num_hidden,
                                       activation=activation,
                                       layer_norm=layer_norm),
                       env,
                       model_scope='reversi_edges')
load_variables(model_load_path, variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reversi_edges'))

# model2 = load_model_2(model2_name, env, other_model_names)

# model1 = RandomAgent()
model2 = RandomAgent()


env.update_opponent_model(model2)


result_list = []
for i in range(num_games):
    state = env.reset()
    done = False
    while not done:
        legal_moves_mask = legal_moves(env.board, 1, env.board_size).astype(bool).reshape((-1, 64))

        if np.random.random() < 0.02:
            action = random_move(legal_moves_mask.reshape((-1, 64)).flatten() / np.sum(legal_moves_mask))
        else:
            q_vals = model1.predict(state)
            q_vals *= legal_moves_mask

            q_vals[~legal_moves_mask] = -np.inf
            row_max = np.max(q_vals, axis=0)
            c_max = np.argmax(row_max)
            r_max = np.argmax(q_vals[:,c_max])
            action = r_max*env.board_size + c_max
        # print(action)
        state, reward, done, _ = env.step(action)
    piece_diff = np.sum(state[:,:,1] - state[:,:,2])
    if piece_diff == 0:
        result_list.append(0)
    else:
        result_list.append(int(np.sign(piece_diff)))

    # print(state[:,:,1] - state[:,:,2])
    if i % 10 == 0:
        print("Wins: ", np.sum(np.array(result_list) == 1))
        print("Ties: ", np.sum(np.array(result_list) == 0))
        print("Number of Games: ", len(result_list))

print("Wins: ", np.sum(np.array(result_list) == 1))
print("Ties: ", np.sum(np.array(result_list) == 0))
print("Number of Games: ", len(result_list))