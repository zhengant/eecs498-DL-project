"""
Collection of functions for initializing the model objects

Depending on how many different models we have/how crazy they get, we could split them into separate files
"""
import numpy as np
import tensorflow as tf

from baselines.common.models import register

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
