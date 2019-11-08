"""
Train and compete new neural nets
"""
import os
import pickle
import sys

from go_game import GoGame
from NNet import NetTrainer
from play import compete
from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()

    NEW_EXAMPLE_PATH = ARGS.new_path
    USED_EXAMPLE_PATH = ARGS.used_path

    if not os.path.exists(USED_EXAMPLE_PATH):
        os.makedirs(USED_EXAMPLE_PATH)

    GAME = GoGame(13, 7.5)

    NEW_NET = NetTrainer(GAME, ARGS)
    NEW_NET.load_checkpoint(ARGS.new_model_path+str(ARGS.type))

    ALL_EXAMPLES = []

    for example_file in os.listdir(NEW_EXAMPLE_PATH):
        with open(NEW_EXAMPLE_PATH + example_file, 'rb') as handle:
            b = pickle.load(handle)
            ALL_EXAMPLES += b['examples']

    if len(ALL_EXAMPLES) == 0:
        print('No Example file found')
        sys.exit()

    NEW_NET.train(ALL_EXAMPLES)
    OLD_NET = NetTrainer(GAME, ARGS)
    OLD_NET.load_checkpoint(ARGS.new_model_path+str(ARGS.type))

    OLD_WIN_COUNT, NEW_WIN_COUNT = compete(OLD_NET, NEW_NET, GAME, ARGS)

    print('OldWins {} NewWins {}'.format(OLD_WIN_COUNT, NEW_WIN_COUNT))

    if NEW_WIN_COUNT > OLD_WIN_COUNT:
        NEW_NET.save_checkpoint(ARGS.new_model_path+str(ARGS.type))
        NEW_NET.save_checkpoint(ARGS.best_model_path+str(ARGS.type))
        for example_file in os.listdir(NEW_EXAMPLE_PATH):
            os.rename(NEW_EXAMPLE_PATH + example_file,
                      USED_EXAMPLE_PATH + example_file)
