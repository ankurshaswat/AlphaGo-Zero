"""
Compete with best model
"""
import os

from go_game import GoGame
from NNet import NetTrainer
from play import compete_random_greedy
from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()
    GAME = GoGame(13, 7.5)

    OLD_WIN_COUNT, NEW_WIN_COUNT, black_win, white_win = compete_random_greedy(GAME, ARGS)

    # if not os.path.exists('../compete_results'):
        # os.makedirs('../compete_results')

    print('Random {} Greedy {}'.format(
        OLD_WIN_COUNT, NEW_WIN_COUNT), flush=True)

    # with open('../compete_results/' + str(ARGS.thread_num) + '.txt', 'w') as file:
        # file.write(str(OLD_WIN_COUNT) + ' ' + str(NEW_WIN_COUNT)+' '+ str(black_win) +' '+ str(white_win))
