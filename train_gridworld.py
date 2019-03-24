import gym
import tensorflow as tf

from environments import MultiItemGridWorld
from single_model_training import dqn


if __name__ == '__main__':
    config = {
        'lr': 1e-2,
        'total_timesteps': 100000,
        'buffer_size': 50000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.02,
        'print_freq': 10,
        'vis': True
    }
    kwargs = {
        'num_layers': 3,
        'num_hidden': 64,
        # 'activation': tf.nn.relu,
        'activation': tf.tanh,
        'layer_norm': False,
    }
    env = MultiItemGridWorld(size=8, num_types=4, rewards=[2, 4, 8, 16],
        empty_prob=0.5, move_penalty=-1, episode_length=32, reward_mask=None)
    dqn(env, 'mlp', config, **kwargs)
