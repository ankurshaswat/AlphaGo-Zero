"""
Compete with best model
"""
import os

from go_game import GoGame
from NNet import NetTrainer
from play import compete
from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()
    GAME = GoGame(13, 7.5)

    NEW_NET = NetTrainer(GAME, ARGS)
    NEW_NET.load_checkpoint(ARGS.temp_model_path+str(ARGS.type))

    OLD_WIN_COUNT, NEW_WIN_COUNT, black_win, white_win = compete(NEW_NET, GAME, ARGS, old_nnet=None)

    if not os.path.exists('../compete_results'):
        os.makedirs('../compete_results')

    print('OldWins {} NewWins {}'.format(
        OLD_WIN_COUNT, NEW_WIN_COUNT), flush=True)

    with open('../compete_results/' + str(ARGS.thread_num) + '.txt', 'w') as file:
        file.write(str(OLD_WIN_COUNT) + ' ' + str(NEW_WIN_COUNT)+' '+ str(black_win) +' '+ str(white_win))
