"""
Collection of environment objects to train on. Probably want these objects to match the openai gym API

We can probably wrap a bunch of openai gym environments with our own stuff to create custom reward functions/outputs

Depending on how many different environments we have/how complicated they get we may want to split these into separate
files
"""

import numpy as np

class MultiItemGridWorld:
    # maybe want separate classes for the distillation environment and the genetic algorithm environment
    def __init__(self, size, num_types, rewards, empty_prob, move_penalty, episode_length):
        """
        init parameters
        :param size: indicates the size of the grid - grid will be size x size
        :param num_types: the number of different items there will be
        :param rewards: rewards[i] should be the reward given for picking up item i
        :param empty_prob: for initialization - the probability that a particular space will be empty
        :param move_penalty: the penalty to be placed on moving
        :param episode_length: number of steps in a single episode
        """
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.size = size
        self.num_types = num_types
        self.rewards = rewards
        self.empty_prob = empty_prob
        self.move_penalty = move_penalty
        self.episode_length = episode_length

        self.current_pos = None

        # reward_mask should be a 1-0 array that tells us which rewards are "on" for a particular episode
        # if reward_mask[i] == 1, then reward for item i is "on"
        self.reward_mask = np.ones(self.num_types)


    def reset(self, reward_mask=None):
        # set grid
        for i in range(self.size):
            for j in range(self.size):
                if np.random.random() < self.empty_prob:
                    self.grid[i,j] = 0
                else:
                    idx = np.random.choice(self.num_types)
                    self.grid[i,j] = idx

        self.reward_mask = reward_mask if reward_mask is not None else np.ones(self.num_types)

        # randomly set starting position
        self.current_pos = np.random.choice(self.size, 2)
        self.grid[self.current_pos[0], self.current_pos[1]] = 0


    def step(self):
        if self.current_pos is None:
            raise RuntimeError('environment not initialized')

