"""
Train and compete new neural nets
"""
import os
import pickle
import sys

from go_game import GoGame
from NNet import NetTrainer
from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()

    NEW_EXAMPLE_PATH = ARGS.new_path
    USED_EXAMPLE_PATH = ARGS.used_path

    if not os.path.exists(USED_EXAMPLE_PATH):
        os.makedirs(USED_EXAMPLE_PATH)

    GAME = GoGame(13, 7.5)

    NEW_NET = NetTrainer(GAME, ARGS)
    NEW_NET.load_checkpoint(ARGS.best_model_path+str(ARGS.type))

    ALL_EXAMPLES = []

    for example_file in os.listdir(NEW_EXAMPLE_PATH):
        with open(NEW_EXAMPLE_PATH + example_file, 'rb') as handle:
            b = pickle.load(handle)
            ALL_EXAMPLES += b['examples']

    if len(ALL_EXAMPLES) == 0:
        print('No Example file found')
        sys.exit()

    NEW_NET.train(ALL_EXAMPLES)
    NEW_NET.save_checkpoint(ARGS.temp_model_path+str(ARGS.type))
