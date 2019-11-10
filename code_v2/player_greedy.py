import os
import random
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

    def get_action(self, cur_state, opponent_action):
        # State is the board

        #Do we have to play oppenent's action ? why don't we get the played board then ?
        #This above step is wasting time.

        #Run the opponent's move
        cur_board = self.game.get_next_state(cur_state, -1 * self.player, opponent_action)

        #Get the possible actions
        possible_actions = self.game.get_valid_moves(cur_board, self.player)

        # ADDITIONAL LOGIC IF POSSIBLE ACTIONS SHAPED INCORRECTLY
    #     # print(actions)
    #     selected_action = None
    #     possible_actions = []
    #     for action , indicator in enumerate(actions):
    #         if indicator == 1:
    #             # selected_action = action
    #             # break
    #             possible_actions.append(action)
        
        
        #can possible actions be null ? or will resign already be in there ?

        high_score = 0
        greedy_action = None

        for action in possible_actions:
            new_board = self.game.get_next_state(cur_board, self.player, action)
            score = self.game.get_score(new_board, self.player) 
            
            if score >= high_score: # Modify if multiple high scores
                greedy_action = action # The greedy action
            
        # assuming best action isn't None and resign and pass will be in the possible actions
        
        #No need for final board
        final_board = self.game.get_next_state(cur_board, self.player, greedy_action)

        return greedy_action




############# IGNORE BELOW

# if __name__ == "__main__":
    #  game = GoGame(13,5.5)

    # board = game.getInitBoard()
    # player = -1

    # for i in range(1000):
    #     actions = game.getValidMoves(board,player)
    #     # print(actions)
    #     selected_action = None
    #     possible_actions = []
    #     for action , indicator in enumerate(actions):
    #         if indicator == 1:
    #             # selected_action = action
    #             # break
    #             possible_actions.append(action)

    #     selected_action = random.choice(possible_actions)        
    #     print(selected_action)
    #     print(board)
    #     print(board.board.get_legal_coords(1))
    #     print(board.board.get_legal_coords(2))
    #     (board,player) = game.getNextState(board,player,selected_action)
