import numpy as np


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


def zero_rewards(reversi_env, prev_board, player, base_reward=0):
    return 0


def simple_rewards(reversi_env, prev_board, player, base_reward=1):
    if not reversi_env.winner == 0:
        if reversi_env.winner * player > 0:
            return base_reward
        elif reversi_env.winner * player < 0:
            return -base_reward
        else:
            return 0


    total = np.sum(reversi_env.board[:,:,player], axis=(0,1))
    total -= np.sum(reversi_env.board[:,:,other_player(player)], axis=(0,1))
    rew = 0
    if total > 0:
        rew = base_reward
    elif total < 0:
        rew = -base_reward
    else:
        rew = 0

    # print(reversi_env.done)
    if not reversi_env.done:
        rew = 0
    # print(reversi_env.board)
    # print(rew)
    return rew


def corner_rewards(reversi_env, prev_board, player, base_reward=1):
    board_diff = reversi_env.board - prev_board
    rew = 0
    if board_diff[0,0,1]:
        rew += 1
    if board_diff[0,-1,1]:
        rew += 1
    if board_diff[-1,0,1]:
        rew += 1
    if board_diff[-1,-1,1]:
        rew += 1
    return rew * base_reward


def edge_rewards(reversi_env, prev_board, player, base_reward=1):
    board_diff = reversi_env.board - prev_board
    rew = 0
    if board_diff[:,0,1].any():
        rew += np.sum(board_diff[:,0,1])
    if board_diff[:,-1,1].any():
        rew += np.sum(board_diff[:,-1,1])
    if board_diff[-1,:,1].any():
        rew += np.sum(board_diff[-1,:,1])
    if board_diff[0,:,1].any():
        rew += np.sum(board_diff[0,:,1])
    return rew * base_reward


def greedy_rewards(reversi_env, prev_board, player, base_reward=1):
    prev_count = np.sum(prev_board[:, :, player] == 1)
    # print(prev_count)
    new_count = np.sum(reversi_env.board[:, :, player] == 1)
    # print(new_count)

    return (new_count - prev_count)*base_reward


def opp_greedy_rewards(reversi_env, prev_board, player, base_reward=1):
    other = other_player(player)

    prev_count = np.sum(prev_board[:, :, other] == 1)
    new_count = np.sum(reversi_env.board[:, :, other] == 1)

    return (prev_count - new_count)*base_reward


def mobility_rewards(reversi_env, prev_board, player, base_reward=1):
    mobility = np.sum(legal_moves(reversi_env.board, player, 8))

    return mobility * base_reward


def opp_mobility_rewards(reversi_env, prev_board, player, base_reward=1):
    opp_mobility = -np.sum(legal_moves(reversi_env.board, other_player(player), 8))

    return opp_mobility * base_reward