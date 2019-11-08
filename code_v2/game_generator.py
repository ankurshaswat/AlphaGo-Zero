"""
Generate Episodes and save in files.
"""

import argparse
import os
import pickle

from play import generate_episodes
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
                        dest="thread_num", type=int, required=True)
    parser.add_argument('-unique_token', action='store',
                        dest="unique_token", required=True)
    parser.add_argument('-numEpisodes', action='store',
                        dest='numEpisodes', type=int, default=1)
    parser.add_argument('-numSimulations', action='store',
                        dest='numSimulations', type=int, default=2)
    parser.add_argument('-cpuct', action='store',
                        dest='cpuct', type=float, default=1.0)
    parser.add_argument('-numStepsForTempChange', action='store',
                        dest='numStepsForTempChange', type=int, default=20)
    parser.add_argument('-netType', action='store',
                        dest='type', type=int, default=3)
    parser.add_argument('-history', action='store_true',
                        dest='history', default=False)
    parser.add_argument('-dropout', action='store',
                        dest='dropout', type=int, default=0.5)
    parser.add_argument('-new_model_path', action='store',
                        dest='new_model_path', default='model.pytorch')
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
    NET.load_checkpoint(str(ARGS.type)+ARGS.new_model_path)

    EPIS = generate_episodes(NET, GAME, ARGS)

    # save epis as pickle using uniqueName_thread
    with open(NEW_EXAMPLE_PATH + UNIQUE_TOKEN+'_'+str(THREAD_NUM)+'_net'+'.pkl', 'wb') as handle:
        pickle.dump({'examples': EPIS}, handle)

    print('Thread Number {} ended and saved file.'.format(THREAD_NUM))
