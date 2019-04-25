import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.deepq.utils import ObservationInput
from reversi_environment import legal_moves
from models import reversi_network
from baselines.common.tf_util import save_variables, load_variables, get_session


def load_model(model_name, env):
    num_layers = 3
    num_hidden = 128
    activation = tf.nn.relu
    layer_norm = False

    model_load_path = "".join(['saved_models/', model_name, '.pkl'])
    model = PredictorAgent(reversi_network(num_layers=num_layers,
                                          num_hidden=num_hidden,
                                          activation=activation,
                                          layer_norm=layer_norm),
                        env,
                        model_scope=model_name)
    load_variables(model_load_path, variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
    return model


class RandomAgent():
    def update(self, *args, **kwargs):
        pass

    def predict(self, board):
        board_size = board.shape[0]
        return np.random.rand(board_size, board_size)


class PlayerAgent():
    def update(self, *args, **kwargs):
        pass

    def predict(self, board):
        print(board[:,:,1] - board[:,:,2])
        q_vals = np.zeros((8,8))

        legal_moves_mask = legal_moves(board, 1, 8).astype(bool)

        row = int(input("Select a row: "))
        col = int(input("Select a col: "))

        while not legal_moves_mask[row][col]:
            print("Illegal Move, please select a legal move!")
            row = int(input("Select a row: "))
            col = int(input("Select a col: "))

        q_vals[row][col] = 1
        return q_vals


class ModelLoader():
    def __init__(self, env, model_scope, other_model_names):
        num_other_models = len(other_model_names)

        self.other_models = []
        for other_model_name in other_model_names:
            model = load_model(other_model_name, env)
            self.other_models.append(model)

    def update(self, network_to_clone_vars):
        pass

    def predict(self, board):
        q_vals_all = []
        for model in self.other_models:
            q_vals = model.predict(np.expand_dims(board, axis=0)).reshape((8,8))
            q_vals_all.append(q_vals)
        q_vals_all = np.stack(q_vals_all, axis=-1)

        board = np.concatenate([board[:,:,0:3], q_vals_all, np.expand_dims(board[:,:,-1], axis=-1)], axis=-1)
        return board


class PredictorAgent():
    def __init__(self, network, env, model_scope):
        with tf.variable_scope(model_scope, reuse=False):
            observation_space = env.observation_space
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)
            self.model_scope = model_scope

    def update(self, network_to_clone_vars):
        pass

    def predict(self, board):
        return self.network(np.expand_dims(board, axis=0))


class CompositeAgent():
    def __init__(self, network, env, model_scope, other_models):
        num_other_models = len(other_models)
        self.other_models = other_models
        self.model_scope = model_scope

        with tf.variable_scope(self.model_scope, reuse=False):
            observation_space = env.observation_space
            observation_space.shape = (8, 8, 4+num_other_models)
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)

    def update(self, network_to_clone_vars):
        pass

    def predict(self, board):
        if not len(self.other_models) == 0:
            q_vals_all = []
            for model in self.other_models:
                q_vals = model.predict(np.expand_dims(board, axis=0)).reshape((8, 8))
                q_vals_all.append(q_vals)
            q_vals_all = np.stack(q_vals_all, axis=-1)
            board = np.concatenate([board[:,:,0:3], q_vals_all, np.expand_dims(board[:,:,-1], axis=-1)], axis=-1)


        return self.network(np.expand_dims(board, axis=0))


class ClonedAgent():
    def __init__(self, network, env, model_scope="cloned_agent", other_models=[]):
        with tf.variable_scope(model_scope, reuse=False):
            observation_space = env.observation_space
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)
            self.model_scope = model_scope
            self.other_models = other_models

    def update(self, network_to_clone_vars):
        this_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope)
        sess = U.get_session()
        sess.run([v_this.assign(v) for v_this, v in zip(this_vars, network_to_clone_vars)])

    def predict(self, board):
        if not len(self.other_models) == 0:
            q_vals_all = []
            for model in self.other_models:
                q_vals = model.predict(np.expand_dims(board, axis=0)).reshape((8, 8))
                q_vals_all.append(q_vals)
            q_vals_all = np.stack(q_vals_all, axis=-1)
            board = np.concatenate([board[:,:,0:3], q_vals_all, np.expand_dims(board[:,:,-1], axis=-1)], axis=-1)


        return self.network(np.expand_dims(board, axis=0))
