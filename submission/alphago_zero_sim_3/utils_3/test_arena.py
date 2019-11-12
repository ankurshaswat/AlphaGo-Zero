import random
from go_game import GoGame
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    game = GoGame(13, 5.5)

    board = game.get_starting_board()
    player = -1

    while True:
        if game.get_game_ended(board, player):
            break

        actions = game.get_valid_moves(board, player)
        actions[-1] = 0
        selected_action = None
        possible_actions = []

        for action, indicator in enumerate(actions):
            if indicator == 1:
                possible_actions.append(action)

        if len(possible_actions) > 0:
            selected_action = random.choice(possible_actions)
        else:
            selected_action = game.get_action_space_size()-1
        board = game.get_next_state(board, player, selected_action)

        print(selected_action)
        board.print_board()

        player = -player
