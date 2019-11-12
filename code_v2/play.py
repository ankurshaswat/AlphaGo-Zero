"""
Generate new episodes of self play.
"""

import time

import numpy as np

from MCT import MCT
# from MCT_orig import MCT


BLACK = -1
WHITE = 1


def generate_episodes(nnet, game, args):  # self play
    """
    Generate game episodes for simulation.
    """

    num_epis = args.numEpisodes
    num_steps_temp_thresh = args.numStepsForTempChange

    # training data, to be filled with (state, action_prob, value) as encountered during the game
    train = []

    mct = MCT(nnet, game, args)

    # mct = MCT(nnet, game, args, greedy=True)

    num_steps = 0

    temp = 1
    for epi in range(num_epis):
        episode = []

        # initial board
        board = game.get_starting_board()
        player = BLACK  # game always starts with BLACK

        start_time = time.time()
        while True:
            # print("Num_steps:",num_steps,flush=True)
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
                # dummy reward for now
                episode.append([t_board, pi_, new_valids, 0])

            # pick action and play
            next_action = np.random.choice(len(action_prob), p=action_prob)

            board = game.get_next_state(board, player, next_action)
            # print(next_action)
            # board.print_board()
            # print(player,next_action,flush=True)
            # update player
            player = -player

            reward = None

            if board.is_terminal():  # maximum steps reached
                reward = game.decide_winner(board, player)
                # reward=-reward #flip reward


            if reward is not None:
                for i in reversed(range(len(episode))):
                    reward = -1 * reward  # flip reward, as player alternates
                    episode[i][-1] = reward
                break

        #print("Episode {}/{} completed".format(epi, num_epis), flush=True)
        print("Episode {}/{} completed in time {:.2f}s".format(epi +
                                                               1, num_epis, time.time()-start_time), flush=True)
        train += episode

    return train


def compete(new_nnet, game, args, old_nnet=None):
    """
    Compete trained NN with old NN
    """
    black_wins = 0
    white_wins = 0
    num_games_per_side = args.numGamesPerSide

    print("num_games_per_side:{}".format(num_games_per_side), flush=True)

    # stores [net,score], score is updated as games are played
    old = [old_nnet, 0]
    new = [new_nnet, 0]

    for game_no in range(num_games_per_side*2):

        if game_no < num_games_per_side:
            player_dict = {BLACK: old, WHITE: new}
        else:
            player_dict = {BLACK: new, WHITE: old}

        black_player = MCT(player_dict[BLACK][0], game, args)
        white_player = MCT(player_dict[WHITE][0], game, args)

        mct_dict = {BLACK: black_player, WHITE: white_player}

        # initial board
        board = game.get_starting_board()

        curr_player = BLACK
        num_steps = 0

        start_time = time.time()

        while True:
            num_steps += 1

            # get action probabilities
            if player_dict[curr_player][0] is None:
                action_prob = np.asarray(game.get_valid_moves(board, curr_player))
                action_prob = (action_prob/np.sum(action_prob)).tolist()
            else:
                action_prob = mct_dict[curr_player].actionProb(
                    board, curr_player, 1)
            # print(curr_player, action_prob, flush=True)
            # pick action and play
            #next_action = np.argmax(action_prob)
            next_action = np.random.choice(len(action_prob), p=action_prob)
            board = game.get_next_state(board, curr_player, next_action)
            # print(curr_player, next_action, flush=True)
            curr_player = -curr_player

            # check if game has ended (or max moves exceeded)
            if board.is_terminal():  # maximum steps reached
                break

        reward = game.decide_winner(board, BLACK)
        if reward == 1:
            black_wins += 1
            player_dict[BLACK][1] += 1
        elif reward == -1:
            white_wins += 1
            player_dict[WHITE][1] += 1
        else:
            player_dict[BLACK][1] += 0.5
            player_dict[WHITE][1] += 0.5

        # print("Old score:{}, New score:{}".format(old[1],new[1]), flush=True)

        print("Old score:{}, New score:{} Time:{:.2f}s Black{} White{}".format(
            old[1], new[1], time.time()-start_time,black_wins,white_wins), flush=True)

    return old[1], new[1],black_wins,white_wins

def compete_random_greedy(game, args):
    """
    Compete trained NN with old NN
    """
    black_wins = 0
    white_wins = 0
    num_games_per_side = args.numGamesPerSide

    print("num_games_per_side:{}".format(num_games_per_side), flush=True)

    # stores [net,score], score is updated as games are played
    old = [0, 0]
    new = [1, 0]

    for game_no in range(num_games_per_side*2):

        if game_no < num_games_per_side:
            player_dict = {BLACK: old, WHITE: new}
        else:
            player_dict = {BLACK: new, WHITE: old}

        if player_dict[BLACK] == new:
            black_player = MCT(player_dict[BLACK][0], game, args,greedy= True)
            white_player = MCT(player_dict[WHITE][0], game, args)
        else:
            black_player = MCT(player_dict[BLACK][0], game, args)
            white_player = MCT(player_dict[WHITE][0], game, args,greedy= True)

        mct_dict = {BLACK: black_player, WHITE: white_player}

        # initial board
        board = game.get_starting_board()

        curr_player = BLACK
        num_steps = 0

        start_time = time.time()

        while True:
            num_steps += 1

            # get action probabilities
            if player_dict[curr_player] == old:
                action_prob = np.asarray(game.get_valid_moves(board, curr_player))
                action_prob = (action_prob/np.sum(action_prob)).tolist()
            else:
                action_prob = mct_dict[curr_player].actionProb(
                    board, curr_player, 1)
            # print(curr_player, action_prob, flush=True)
            # pick action and play
            #next_action = np.argmax(action_prob)
            next_action = np.random.choice(len(action_prob), p=action_prob)
            board = game.get_next_state(board, curr_player, next_action)
            # print(curr_player, next_action, flush=True)
            curr_player = -curr_player

            # check if game has ended (or max moves exceeded)
            if board.is_terminal():  # maximum steps reached
                break

        reward = game.decide_winner(board, BLACK)
        if reward == 1:
            black_wins += 1
            player_dict[BLACK][1] += 1
        elif reward == -1:
            white_wins += 1
            player_dict[WHITE][1] += 1
        else:
            player_dict[BLACK][1] += 0.5
            player_dict[WHITE][1] += 0.5

        # print("Old score:{}, New score:{}".format(old[1],new[1]), flush=True)

        print("Old score:{}, New score:{} Time:{:.2f}s Black{} White{}".format(
            old[1], new[1], time.time()-start_time,black_wins,white_wins), flush=True)

    return old[1], new[1],black_wins,white_wins
