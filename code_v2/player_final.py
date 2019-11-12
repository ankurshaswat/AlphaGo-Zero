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

        

    #     # Statse is the board

    #     #Do we have to play oppenent's action ? why don't we get the played board then ?
    #     #This above step is wasting time.

    #     #Run the opponent's move
    #     cur_board = self.game.get_next_state(cur_state, -1 * self.player, opponent_action)

    #     #Get the possible actions
    #     possible_actions = self.game.get_valid_moves(cur_board, self.player)

    #     # ADDITIONAL LOGIC IF POSSIBLE ACTIONS SHAPED INCORRECTLY
    # #     # print(actions)
    # #     selected_action = None
    # #     possible_actions = []
    # #     for action , indicator in enumerate(actions):
    # #         if indicator == 1:
    # #             # selected_action = action
    # #             # break
    # #             possible_actions.append(action)
        
        
    #     #can possible actions be null ? or will resign already be in there ?

    #     high_score = 0
    #     greedy_action = None

    #     for action in possible_actions:
    #         new_board = self.game.get_next_state(cur_board, self.player, action)
    #         score = self.game.get_score(new_board, self.player) 
            
    #         if score >= high_score: # Modify if multiple high scores
    #             greedy_action = action # The greedy action
            
    #     # assuming best action isn't None and resign and pass will be in the possible actions
        
    #     #No need for final board
    #     final_board = self.game.get_next_state(cur_board, self.player, greedy_action)

        return action

