import numpy as np


def other_player(current_player):
    return 3 - current_player


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
    if board_diff[0][0][1] or board_diff[0][-1][1] or board_diff[-1][0][1] or board_diff[-1][-1][1]:
        rew = base_reward
    return rew


def greedy_rewards(reversi_env, prev_board, player, base_reward=1):
    prev_count = np.sum(prev_board[:, :, player] == 1)
    print(prev_count)
    new_count = np.sum(reversi_env.board[:, :, player] == 1)
    print(new_count)

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