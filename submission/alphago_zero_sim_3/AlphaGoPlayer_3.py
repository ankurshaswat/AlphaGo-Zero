import sys

sys.path.append('utils_3/')

import os
import random
from MCT import MCT
from go_game import GoGame
from NNet import NetTrainer
from play import compete
import argparse
import numpy as np

# CHECK BY RUNNING THIS ? AND CHANGE THE IMPORT STATEMENTS

def parse_args():
    """
    Argument Parser.
    """
    parser = argparse.ArgumentParser(description='Generate New Games.')
    #parser.add_argument('-cuda', action='store_true', default=False)
    parser.add_argument('-cuda', action='store_true', default=True)
    parser.add_argument('-new_path', action='store',
                        dest="new_path", default='../new_examples/')
    parser.add_argument('-used_path', action='store',
                        dest="used_path", default='../used_examples/')
    parser.add_argument('-thread_num', action='store',
                        dest="thread_num", type=int, default=0)
    parser.add_argument('-unique_token', action='store',
                        dest="unique_token", default='notSpecified')
    parser.add_argument('-numEpisodes', action='store',
                        dest='numEpisodes', type=int, default=100)#10
    parser.add_argument('-numSimulations', action='store',
                        dest='numSimulations', type=int, default=100)#100
    parser.add_argument('-cpuct', action='store',
                        dest='cpuct', type=float, default=1.0)
    parser.add_argument('-numStepsForTempChange', action='store',
                        dest='numStepsForTempChange', type=int, default=30)
    parser.add_argument('-netType', action='store',
                        dest='type', type=int, default=3)
    parser.add_argument('-history', action='store_true',
                        dest='history', default=False)
    parser.add_argument('-dropout', action='store',
                        dest='dropout', type=float, default=0.2)
    parser.add_argument('-lr', action='store',
                        dest='lr', type=float, default=0.001)
    parser.add_argument('-momentum', action='store',
                        dest='momentum', type=float, default=0.9)
    parser.add_argument('-l2_regularization', action='store',
                        dest='l2_regularization', type=float, default=1e-5)#1e-5
    parser.add_argument('-epochs', action='store',
                        dest='epochs', type=int, default=10)#10
    parser.add_argument('-batch_size', action='store',
                        dest='batch_size', type=int, default=64)
    parser.add_argument('-numGamesPerSide', action='store',
                        dest='numGamesPerSide', type=int, default=2)#10
    parser.add_argument('-best_model_path', action='store',
                        dest='best_model_path', default='model_3/model.pytorch3')
    parser.add_argument('-temp_model_path', action='store',
                        dest='temp_model_path', default='model_3/model.pytorch3')
    args = parser.parse_args([])
    return args


class AlphaGoPlayer():
    def __init__(self, _, seed, player):
        self.game = GoGame(13,7.5) # THe Go Game class

        #THERE IS NO init state
        self.board = self.game.get_starting_board()

        self.seed = seed

        self.player = -1 if player == 1 else 1

        self.args = parse_args()
        self.nnet = NetTrainer(self.game, self.args)
        self.nnet.load_checkpoint(self.args.best_model_path)

        self.mct = MCT(self.nnet, self.game, self.args,noise=False)
        


    def get_action(self, _, opponent_action):

        if opponent_action != -1 : # MEANS 
            self.board = self.game.get_next_state(self.board, -1 * self.player, opponent_action)

        self.board.set_move_num(0)
        action_probs = self.mct.actionProb(self.board, self.player, 0)
        self.board.set_move_num(-1)

        best_action = np.argmax(action_probs)
        self.board = self.game.get_next_state(self.board,self.player,best_action)
        
        return best_action
        
