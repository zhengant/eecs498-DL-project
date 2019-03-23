"""
Reversi Environment

Players are represented by 1 and -1.
"""

import numpy as np


def simple_rewards(board, player, winner=0):
    base_reward = 1
    if not winner == 0:
        if winner * player > 0:
            return base_reward
        elif winner * player < 0:
            return -base_reward
        else:
            return 0

    total = np.sum(board, axis=(0,1))
    if total * player > 0:
        return base_reward
    elif total * player < 0:
        return -base_reward
    else:
        return 0


class ReversiEnvironment:
    def __init__(self, board_size=8, reward_fn=simple_rewards):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.reward_fn = reward_fn
        self.done = False
        self.winner = 0
        self.modify_pieces = []

        self.reset()

    def step(self, action, player):
        # Trying to step finished board
        if self.done:
            reward = self.reward_fn(self.board, player, self.winner)
            return self.board, reward, self.done, None

        x, y = action
        reward = 0

        # Check for Illegal Move
        self.modify_pieces = []
        if self.check_for_illegal_move(x, y, player):
            # Illegal Move
            self.done = True
            self.winner = -player
            reward = self.reward_fn(self.board, player, self.winner)
            return self.board, reward, self.done, None
        else:
            # Legal Move
            for x_loc, y_loc in self.modify_pieces:
                self.board[x_loc, y_loc] = player
            self.board[x,y] = player
            reward = self.reward_fn(self.board, player)

        # Check for full board
        if np.sum(np.abs(self.board), axis=(0,1)) == self.board_size*self.board_size:
            self.done = True
            return self.board, reward, self.done, None

        return self.board, 0, self.done, None

    def reset(self):
        self.board = np.zeros((self.board_size,self.board_size))
        self.board[self.board_size//2-1,self.board_size//2-1] = 1
        self.board[self.board_size//2,self.board_size//2] = 1
        self.board[self.board_size//2-1,self.board_size//2] = -1
        self.board[self.board_size//2,self.board_size//2-1] = -1
        self.done = False
        self.winner = 0
        self.modify_pieces = []

    def check_for_illegal_move(self, x_action, y_action, player):
        step_sizes = np.arange(-1, 2)
        for x_diff in step_sizes:
            for y_diff in step_sizes:
                if x_diff == 0 and y_diff == 0:
                    if not self.board[x_action, y_action] == 0:
                        self.modify_pieces = []
                        return True;
                else:
                    x = x_action + x_diff
                    y = y_action + y_diff
                    if self.is_inside_board(x,y) and self.board[x, y] == -player:
                        self.get_pieces_to_modify(x, y, x_diff, y_diff, player)
        if not self.modify_pieces:
            return True
        else:
            return False

    def get_pieces_to_modify(self, x, y, x_diff, y_diff, player):
        potential_pieces_to_modify = []
        while self.is_inside_board(x, y) and self.board[x,y] == -player:
            potential_pieces_to_modify.append((x,y))
            x += x_diff
            y += y_diff
        if self.is_inside_board(x,y) and self.board[x,y] == player:
            self.modify_pieces += potential_pieces_to_modify

    def is_inside_board(self, x, y):
        if x >= self.board_size or x < 0:
            return False
        if y >= self.board_size or y < 0:
            return False
        return True

# env = ReversiEnvironment(board_size=6)
# print(env.board)
#
# def make_move(x,y, player):
#     board, rew, done, info = env.step((x, y), player)
#     print(board)
#     print("Reward:", rew)
#     print("Done:", done)
#
# make_move(1,3,1)
# make_move(3,4,-1)
# make_move(4,3,1)
# make_move(5,2,-1)
# make_move(4,2,1)
# make_move(1,2,-1)
# make_move(5,3,1)
# make_move(5,4,-1)
# make_move(1,1,1)
# make_move(0,2,-1)
# make_move(3,1,1)
# make_move(3,0,-1)
# make_move(4,4,1)
# make_move(0,3,-1)
# make_move(2,4,1)
# make_move(1,5,-1)
# make_move(2,5,1)
# make_move(0,0,-1)
# make_move(0,1,1)
# make_move(3,5,-1)
# make_move(2,1,1)
# make_move(2,0,-1)
# make_move(1,0,1)
# make_move(1,4,1)
# make_move(0,5,-1)
# make_move(0,4,1)
# make_move(4,1,1)
# make_move(5,0,-1)
# make_move(4,0,1)
# make_move(5,1,1)
# make_move(5,5,1)
# make_move(4,5,-1)
# make_move(1326,1261,1)