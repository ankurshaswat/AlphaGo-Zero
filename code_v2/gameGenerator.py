"""
Generate Episodes and save in files.
"""

import argparse
import os
import pickle

from Generator import generateEpisodes
from NNet import NetTrainer
from go_game import GoGame


def parse_args():
    """
    Argument Parser.
    """
    parser = argparse.ArgumentParser(description='Generate New Games.')

    parser.add_argument('-cuda', action='store_true', default=False)
    parser.add_argument('-new_path', action='store',
                        dest="new_path", default='../new_examples')
    parser.add_argument('-thread_num', action='store',
                        dest="thread_num", type=int)
    parser.add_argument('-unique_token', action='store',
                        dest="unique_token")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    ARGS = parse_args()

    NEW_EXAMPLE_PATH = ARGS.new_path

    if not os.path.exists(NEW_EXAMPLE_PATH):
        os.makedirs(NEW_EXAMPLE_PATH)

    THREAD_NUM = ARGS.thread_num
    UNIQUE_TOKEN = ARGS.unique_token

    GAME = GoGame(13, 7.5)

    NET = NetTrainer(GAME, ARGS)
    NET.load_checkpoint('model.pytorch')

    EPIS = generateEpisodes(NET, ARGS)

    # save epis as pickle using uniqueName_thread
    with open(NEW_EXAMPLE_PATH + UNIQUE_TOKEN+'_'+str(THREAD_NUM)+'.pkl', 'wb') as handle:
        pickle.dump({'examples': EPIS}, handle)

    print('Thread Number {} ended and saved file.'.format(THREAD_NUM))
