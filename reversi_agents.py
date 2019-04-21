import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.deepq.utils import ObservationInput
from reversi_environment import legal_moves


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


class ClonedAgent():
    def __init__(self, network, env):
        with tf.variable_scope("cloned_agent", reuse=False):
            observation_space = env.observation_space
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)

    def update(self, network_to_clone_vars):
        this_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cloned_agent")
        sess = U.get_session()
        sess.run([v_this.assign(v) for v_this, v in zip(this_vars, network_to_clone_vars)])

    def predict(self, board):
        with tf.variable_scope("cloned_agent", reuse=False):
            return self.network(np.expand_dims(board, axis=0))