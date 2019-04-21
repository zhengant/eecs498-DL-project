"""
Collection of functions for initializing the model objects

Depending on how many different models we have/how crazy they get, we could split them into separate files
"""
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc

from baselines.common.models import register

from reversi_environment import legal_moves

# # Example model in baselines
# @register("cnn_small")
# def cnn_small(**conv_kwargs):
#     def network_fn(X):
#         h = tf.cast(X, tf.float32) / 255.

#         activ = tf.nn.relu
#         h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
#         h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
#         h = conv_to_fc(h)
#         h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
#         return h
#     return network_fn

# Template for defining our own network.
@register('network_name')
def network():
    def network_fn(X):
        return X

    return network_fn


@register('reversi_network')
def reversi_network(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    def network_fn(X):
        board = X[:,:,:,0:3]
        # legal_mask = X[:,:,:,3]

        h = tf.layers.flatten(board)
        for i in range(num_layers-1):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)
        h = fc(h, 'mlp_fc{}'.format(num_layers-1), nh=64, init_scale=np.sqrt(2))
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
        h = activation(h)

        # board_size = board.shape[1]
        # h = tf.boolean_mask(h, tf.not_equal(h, tf.constant(0, dtype=tf.float32)))

        return h

    return network_fn
