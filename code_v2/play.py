"""
Generate new episodes of self play.
"""

import numpy as np

from MCT import MCT

BLACK = -1
WHITE = 1
MAX_STEPS = 13*13


def generate_episodes(nnet, game, args):  # self play
    """
    Generate game episodes for simulation.
    """

    num_epis = args.numEpisodes
    num_steps_temp_thresh = args.numStepsForTempChange

    # training data, to be filled with (state, action_prob, value) as encountered during the game
    train = []

    # initial board
    board = game.get_starting_board()

    mct = MCT(nnet, game, args)

    num_steps = 0

    player = BLACK  # game always starts with BLACK

    temp = 1
    for _ in range(num_epis):
        episode = []
        while True:
            num_steps += 1
            if num_steps >= num_steps_temp_thresh:
                temp = 0

            # get action probabilities
            action_prob = mct.actionProb(board, player, temp)

            # get valid actions
            valids = game.get_valid_moves(board, player)

            # append board(randomly flipped/rotated),action_prob
            # to training data (value/reward will be added later)

            sym = game.get_symmetries(
                board, action_prob, valids, player, history=False)
            for t_board, pi_, new_valids in sym:
                episode.append([t_board, pi_, new_valids, 0])  # dummy reward for now

            # pick action and play
            next_action = np.random.choice(len(action_prob), p=action_prob)

            board = game.get_next_state(board, player, next_action)

            # update player
            player = -player

            reward = None

            if(num_steps >= MAX_STEPS) or board.is_terminal():  # maximum steps reached
                reward = game.decide_winner(board, player)

            if reward is not None:
                for i in reversed(range(len(train))):
                    reward = -1 * reward  # flip reward, as player alternates
                    episode[i][-1] = reward
                break

        train += episode

    return train
