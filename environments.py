"""
Collection of environment objects to train on. Probably want these objects to match the openai gym API

We can probably wrap a bunch of openai gym environments with our own stuff to create custom reward functions/outputs

Depending on how many different environments we have/how complicated they get we may want to split these into separate
files
"""

import numpy as np
import gym

class MultiItemGridWorld:
    # maybe want separate classes for the distillation environment and the genetic algorithm environment
    def __init__(self, size, num_types, rewards, empty_prob, move_penalty, episode_length, reward_mask):
        """
        init parameters
        :param size: indicates the size of the grid - grid will be size x size
        :param num_types: the number of different items there will be
        :param rewards: rewards[i] should be the reward given for picking up item i
        :param empty_prob: for initialization - the probability that a particular space will be empty
        :param move_penalty: the penalty to be placed on moving. Penalty will be added to reward so it should be signed
        appropriately
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
        self.current_step = -1

        # reward_mask should be a 1-0 array that tells us which rewards are "on" for a particular episode
        # if reward_mask[i] == 1, then reward for item i is "on"
        self.reward_mask = reward_mask if reward_mask is not None else np.ones(self.num_types)

        self.action_map = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([0, 1]),
            3: np.array([1, 0]),
            4: np.array([0, -1])
        }

        self.observation_space = gym.spaces.Box(low=-2, high=self.num_types-1, shape=(self.size, self.size), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(len(self.action_map))


    def reset(self, reward_mask=None):
        # set grid
        for i in range(self.size):
            for j in range(self.size):
                if np.random.random() < self.empty_prob:
                    self.grid[i,j] = -1
                else:
                    idx = np.random.choice(self.num_types)
                    self.grid[i,j] = idx

        # randomly set starting position
        self.current_pos = np.random.choice(self.size, 2)
        self.grid[self.current_pos[0], self.current_pos[1]] = -2

        self.current_step = 0

        return (self.grid, self.current_pos)


    def step(self, action):
        """
        Make an action
        :param action: 0 - stay, 1 - north, 2 - east, 3 - south, 4 - west
        :return: state, reward, done, info
        """
        if self.current_pos is None:
            raise RuntimeError('environment not initialized')
        if self.current_step >= self.episode_length:
            raise RuntimeError('episode ended')

        else:


            reward = 0

            # make current position empty
            self.grid[self.current_pos[0], self.current_pos[1]] = -1

            # update position
            self.current_pos = np.clip(self.current_pos + self.action_map[action], 0, self.size-1)
            item = self.grid[self.current_pos[0], self.current_pos[1]]
            # mark current_position on the board
            self.grid[self.current_pos[0], self.current_pos[1]] = -2

            if item >= 0:
                reward += np.multiply(self.rewards, self.reward_mask)[item]

            if not action == 0:
                reward += self.move_penalty

            self.current_step += 1

            return self.grid, reward, self.current_step >= self.episode_length, 0



