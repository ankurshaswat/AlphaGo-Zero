import os

from go_game import GoGame
from NNet import NetTrainer
from play import compete
from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()
    GAME = GoGame(13, 7.5)

    OLD_NET = NetTrainer(GAME, ARGS)
    OLD_NET.load_checkpoint(ARGS.new_model_path+str(ARGS.type))

    NEW_NET = NetTrainer(GAME, ARGS)
    NEW_NET.load_checkpoint(ARGS.temp_model_path+str(ARGS.type))

    OLD_WIN_COUNT, NEW_WIN_COUNT = compete(OLD_NET, NEW_NET, GAME, ARGS)

    if not os.path.exists('../compete_results'):
        os.makedirs('../compete_results')

    print('ThreadNUm {} OldWins {} NewWins {}'.format(
        ARGS.thread_num, OLD_WIN_COUNT, NEW_WIN_COUNT))
    with open('../compete_results/' + ARGS.thread_num + '.txt', 'w') as file:
        file.write(str(OLD_WIN_COUNT) + ' ' + str(NEW_WIN_COUNT))
