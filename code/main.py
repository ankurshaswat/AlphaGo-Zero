"""
Main file to load/create model and start training.
Created by ankurshaswat(Shashwat Shivam) on 22/10/16.
"""

import argparse
import logging
import os
import time


def parse_args():
    """
    Define and parse arguments passed on calling script
    """
    arg_parser = argparse.ArgumentParser()

    # TODO : Add arguments here.

    args = vars(arg_parser.parse_args())

    return args


if __name__ == "__main__":
    ARGS = parse_args()

    LOGS_PATH = '../LOGS'

    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

    logging.basicConfig(
        filename=LOGS_PATH+'/'+TIMESTAMP+"_main.log",
        level=logging.INFO,
    )

    # Start code after this.

    # TODO : Trainer Class
    # trainer = Trainer()

    # TODO : Write game board Class
    # game_board = Game()

    # TODO : Write NeuralNet Class
    # neural_net = Net()

    if ARGS['load_model']:
        # TODO : Add load_model function
        # neural_net.load_model(ARGS['model_path'])
        pass

    while True:
        # TODO : Generate Games
        # TODO : Start Training
        pass
