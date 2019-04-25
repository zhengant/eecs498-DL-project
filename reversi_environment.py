"""
Reversi Environment

Players are represented by 1 and -1.
"""

import numpy as np
import gym.spaces
from reversi_rewards import simple_rewards

np.set_printoptions(precision=3)

def other_player(current_player):
    return 3 - current_player


def legal_moves(board, player, board_size, default_value=0):
    legal_move_mask = default_value * np.ones((board_size, board_size))
    for r in range(board_size):
        for c in range(board_size):
            if not check_for_illegal_move(board, r, c, player, board_size) == True:
                legal_move_mask[r,c] = 1
    return legal_move_mask


def check_for_illegal_move(board, x_action, y_action, player, board_size, verbose=False):
    pieces_to_modify = []
    step_sizes = np.arange(-1, 2)
    for x_diff in step_sizes:
        for y_diff in step_sizes:
            # Looking in all 8 directions and checking the placed piece
            if x_diff == 0 and y_diff == 0:
                # Can't place a piece on a filled spot
                if not board[x_action, y_action, 0] == 1:
                    return True
            else:
                x = x_action + x_diff
                y = y_action + y_diff
                if verbose:
                    print(x_diff, y_diff)
                if is_inside_board(x, y, board_size) and board[x, y, other_player(player)] == 1:
                    pieces_to_modify += get_pieces_to_modify(board, x, y, x_diff, y_diff, player, board_size, verbose)
            if verbose:
                print(pieces_to_modify)
    if not pieces_to_modify:
        return True
    else:
        return pieces_to_modify


def get_pieces_to_modify(board, x, y, x_diff, y_diff, player, board_size, verbose=False):
    potential_pieces_to_modify = []
    # Find a string of pieces to flip
    while is_inside_board(x, y, board_size) and board[x, y, other_player(player)] == 1:
        potential_pieces_to_modify.append((x,y))
        x += x_diff
        y += y_diff
    if verbose:
        print(board)
        print(potential_pieces_to_modify)
    # Check if the end of the string is one of the current player's pieces
    if is_inside_board(x, y, board_size) and board[x,y, player] == 1:
        return potential_pieces_to_modify
    return []


def is_inside_board(x, y, board_size):
    if x >= board_size or x < 0:
        return False
    if y >= board_size or y < 0:
        return False
    return True


def random_move(legal_moves_board,):
    out = np.random.multinomial(1, legal_moves_board.flatten(), 1)
    return np.argmax(out)


class ReversiEnvironment:
    def __init__(self, opponent_model=None, board_size=8, reward_fn=simple_rewards, base_reward=1):
        self.opponent_model = opponent_model
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size, 3))
        self.reward_fn = reward_fn
        self.base_reward = 1
        self.done = False
        self.winner = 0
        self.starting_player = -1

        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(self.board_size, self.board_size, 4),
                                                dtype=np.int32)

        self.action_space = gym.spaces.Discrete(self.board_size*self.board_size)

    def step(self, action):
        if not np.sum(self.board, axis=2).all() == 1:
            raise("multiple states for board space")

        prev_board = self.board.copy()
        x = action // self.board_size
        y = action % self.board_size
        # print(x, y)
        player = 1
        reward = 0

        # Trying to step finished board
        if self.done:
            reward = self.reward_fn(self, prev_board, player, base_reward=self.base_reward)
            # print(legal_moves(self.next_board, player, self.board_size))
            return np.concatenate((self.board, np.expand_dims(legal_moves(self.board, player, self.board_size), axis=-1)),  axis=-1), reward, self.done, None

        num_legal_moves = np.sum(legal_moves(self.board, 1, self.board_size)>0, axis=(0,1))
        if num_legal_moves > 0:
            # Check for Illegal Move
            modify_pieces = check_for_illegal_move(self.board, x, y, player, self.board_size, verbose=False)
            if modify_pieces == True:
                # Illegal Move
                # self.done = True
                # self.winner = -player
                # reward = -self.base_reward
                reward = 0
                # print(legal_moves(self.board, player, self.board_size))
                print("illegal")
                print(x, y)
                # exit(1)
                return np.concatenate((self.board, np.expand_dims(legal_moves(self.board, player, self.board_size), axis=-1)),  axis=-1), reward, self.done, None
            else:
                # Legal Move
                for x_loc, y_loc in modify_pieces:
                    self.board[x_loc, y_loc, :] = 0
                    self.board[x_loc, y_loc, player] = 1
                self.board[x,y, :] = 0
                self.board[x,y, player] = 1
                reward = self.reward_fn(self, prev_board, player, base_reward=self.base_reward)

            self.opponent_move()

            # Check for full board
            if np.sum(np.abs(self.board[:,:,1:3]), axis=(0,1,2)) == self.board_size*self.board_size:
                self.done = True
                reward = self.reward_fn(self, prev_board, player, base_reward=self.base_reward)
                # print(legal_moves(self.next_board, player, self.board_size))
                return np.concatenate((self.board, np.expand_dims(legal_moves(self.board, player, self.board_size), axis=-1)),  axis=-1), reward, self.done, None
        else:
            if not self.opponent_move():
                self.done = True
                reward = self.reward_fn(self, prev_board, player, base_reward=self.base_reward)
                # print(legal_moves(self.next_board, player, self.board_size))
                return np.concatenate((self.board, np.expand_dims(legal_moves(self.board, player, self.board_size), axis=-1)),  axis=-1), reward, self.done, None

        # print(legal_moves(self.next_board, player, self.board_size))
        return np.concatenate((self.board, np.expand_dims(legal_moves(self.board, player, self.board_size), axis=-1)),  axis=-1), reward, self.done, None

    def reset(self):
        self.board = np.zeros((self.board_size,self.board_size, 3))
        self.board[:,:,0] = 1
        self.board[self.board_size//2-1,self.board_size//2-1, 0] = 0
        self.board[self.board_size//2,self.board_size//2, 0] = 0
        self.board[self.board_size//2-1,self.board_size//2, 0] = 0
        self.board[self.board_size//2,self.board_size//2-1, 0] = 0
        self.done = False
        self.winner = 0
        self.modify_pieces = []
        self.starting_player = -1

        if self.starting_player == -1:
            self.board[self.board_size//2-1,self.board_size//2-1, 2] = 1
            self.board[self.board_size//2,self.board_size//2, 2] = 1
            self.board[self.board_size//2-1,self.board_size//2, 1] = 1
            self.board[self.board_size//2,self.board_size//2-1, 1] = 1
            self.opponent_move()
        else:
            self.board[self.board_size // 2 - 1, self.board_size // 2 - 1, 1] = 1
            self.board[self.board_size // 2, self.board_size // 2, 1] = 1
            self.board[self.board_size // 2 - 1, self.board_size // 2, 2] = 1
            self.board[self.board_size // 2, self.board_size // 2 - 1, 2] = 1

        player = 1
        # if self.starting_player == -1:
        #     player = 2
        self.next_board = self.board.copy()

        return np.concatenate((self.board, np.expand_dims(legal_moves(self.board, player, self.board_size), axis=-1)),  axis=-1)


    def opponent_move(self):
        self.next_board = self.board.copy()
        player = 2
        transformed_board = self.board.copy()
        transformed_board[:,:,1], transformed_board[:,:,2] = transformed_board[:,:,2].copy(), transformed_board[:,:,1].copy()

        legal_moves_mask = legal_moves(transformed_board, 1, self.board_size).astype(bool)
        if np.sum(legal_moves_mask>0, axis=(0,1)) == 0:
            return False

        if np.random.random() < 0.02:
            action = random_move(legal_moves_mask.reshape((-1, 64)).flatten() / np.sum(legal_moves_mask))
            # print(action)
            r_max = action // self.board_size
            c_max = action % self.board_size

        # print(transformed_board[:,:,1] - transformed_board[:,:,2])
        # print(legal_moves_mask)
        else:
            Q_vals = self.opponent_model.predict(np.concatenate((transformed_board, np.expand_dims(legal_moves(transformed_board, 1, self.board_size), axis=-1)),  axis=-1)).reshape((8,8))
            Q_vals *= legal_moves_mask
            Q_vals[~legal_moves_mask] = -np.inf
            row_max = np.max(Q_vals, axis=0)
            c_max = np.argmax(row_max)
            r_max = np.argmax(Q_vals[:,c_max])


        # print(self.board[:,:,1] - self.board[:,:,2])
        # print(r_max, c_max)

        # Check for Illegal Move
        # print(r_max, c_max)
        modify_pieces = check_for_illegal_move(self.board, r_max, c_max, player, self.board_size)
        if not modify_pieces == True:
            # Legal Move
            for x_loc, y_loc in modify_pieces:
                self.board[x_loc, y_loc, :] = 0
                self.board[x_loc, y_loc, player] = 1
            self.board[r_max, c_max, :] = 0
            self.board[r_max, c_max, player] = 1
        else:
            print("illegal")
        return True


    def update_opponent_model(self, opponent_model):
        self.opponent_model = opponent_model


    # Stores the pieces that need to be modified

# env = ReversiEnvironment(board_size=6)
# print(env.board)
#
# def make_move(x,y, player):
#     board, rew, done, info = env.step((x, y, player))
#     print(board)
#     print("Reward:", rew)
#     print("Done:", done)
#
# player = 1
# print(legal_moves(env, player))
# make_move(1,3,player)
#
# player *= -1
# print(legal_moves(env, player))
# make_move(3,4,player)
#
# player *= -1
# print(legal_moves(env, player))
# make_move(4,3,player)

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