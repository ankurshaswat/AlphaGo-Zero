"""
Generate Episodes and save in files.
"""

import os
import pickle

from go_game import GoGame
from NNet import NetTrainer
from play import generate_episodes
from utils import parse_args

if __name__ == "__main__":

    ARGS = parse_args()

    NEW_EXAMPLE_PATH = ARGS.new_path

    if not os.path.exists(NEW_EXAMPLE_PATH):
        os.makedirs(NEW_EXAMPLE_PATH)

    THREAD_NUM = ARGS.thread_num
    UNIQUE_TOKEN = ARGS.unique_token

    GAME = GoGame(13, 7.5)

    NET = NetTrainer(GAME, ARGS)
    NET.load_checkpoint(ARGS.new_model_path+str(ARGS.type))

    EPIS = generate_episodes(NET, GAME, ARGS)

    # save epis as pickle using uniqueName_thread
    with open(NEW_EXAMPLE_PATH + UNIQUE_TOKEN+'_'+str(THREAD_NUM)+'_net'+'.pkl', 'wb') as handle:
        pickle.dump({'examples': EPIS}, handle)

    print('Thread Number {} ended and saved file.'.format(THREAD_NUM))
