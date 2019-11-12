import os
import random
from MCT import MCT
from go_game import GoGame
from NNet import NetTrainer
from play import compete
from utils import parse_args

import numpy as np


class AlphaGoPlayer():
    def __init__(self, init_state, seed, player):
        self.game = GoGame(init_state) # THe Go Game class

        #THERE IS NO init state

        # self.init_state = init_state
        self.seed = seed
        self.player = player

        # WHERE ARE WE GETTING THE ARGS FROM ? 
        self.args = parse_args()
        self.nnet = NetTrainer(self.game, self.args)
        self.nnet.load_checkpoint(self.args.best_model_path+str(self.args.type))

        self.mct = MCT(self.nnet, self.game, self.args)
        


    def get_action(self, cur_state, opponent_action):


        cur_board = self.game.get_next_state(cur_state, -1 * self.player, opponent_action)

        action_probs = self.mct.actionProb(cur_board, self.player, 0)

        best_action = np.argmax(action_probs)
        
        return best_action
        
