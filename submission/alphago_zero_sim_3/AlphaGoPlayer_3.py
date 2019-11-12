import sys

sys.path.append('utils_3/')

import os
import random
from MCT import MCT
from go_game import GoGame
from NNet import NetTrainer
from play import compete
from utils import parse_args

import numpy as np

# CHECK BY RUNNING THIS ? AND CHANGE THE IMPORT STATEMENTS

current_args = dotdict({
'cuda' : True,
'new_path' :  '../new_examples/',
'used_path' :  '../used_examples/',
'thread_num' :  0,
'unique_token' :  'notSpecified',
'numEpisodes' :  10,
'numSimulations' :  100,
'cpuct' :  1.0,
'numStepsForTempChange' :  30,
'type' :  3,
'history' :  False,
'dropout' :  0.2,
'lr' :  0.005,
'momentum' :  0.9,
'l2_regularization' :  1e-5,
'epochs' :  20,
'batch_size' :  64,
'numGamesPerSide' :  2,
'best_model_path' :  'model_3/model.pytorch3'
})



class AlphaGoPlayer():
    def __init__(self, _, seed, player):
        self.game = GoGame(13,7.5) # THe Go Game class

        #THERE IS NO init state
        self.board = self.game.get_starting_board()

        self.seed = seed

        self.player = -1 if player == 1 else 1

        self.args = current_args
        self.nnet = NetTrainer(self.game, self.args)
        self.nnet.load_checkpoint(self.args.best_model_path)

        self.mct = MCT(self.nnet, self.game, self.args)
        


    def get_action(self, _, opponent_action):

        if opponent_action != -1 : # MEANS 
            self.board = self.game.get_next_state(self.board, -1 * self.player, opponent_action)

        self.board.set_move_num(0)
        action_probs = self.mct.actionProb(self.board, self.player, 0,noise=False)
        self.board.set_move_num(-1)

        best_action = np.argmax(action_probs)
        self.board = self.game.get_next_state(self.board,self.player,best_action)
        
        return best_action
        
