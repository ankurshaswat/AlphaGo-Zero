"""
Create a random new net.
"""
import os

from go_game import GoGame
from NNet import NetTrainer
from utils import parse_args

if __name__ == "__main__":

    ARGS = parse_args()

    GAME = GoGame(13, 7.5)

    if not os.path.exists('../models'):
        os.makedirs('../models')

    NET = NetTrainer(GAME, ARGS)
    NET.save_checkpoint(ARGS.best_model_path+str(ARGS.type))
