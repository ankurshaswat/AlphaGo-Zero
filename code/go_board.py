"""
Wrapper over Board class of pachiy_py
"""

import numpy as np
import pachi_py

BLACK = np.array([1, 0, 0])
WHITE = np.array([0, 1, 0])
EMPTY = np.array([0, 0, 1])


class GoBoard():
    """
    Wrapper over board class to modify representations and return new copies.
    """

    def __init__(self, board_size, board=None):
        self.board_size = board_size
        if board is None:
            self.board = pachi_py.CreateBoard(board_size)
        else:
            self.board = board

    def get_pass_action(self):
        """
        Get representation for pass action
        """

        return self.board_size**2

    def get_resign_action(self):
        """
        Get representation for resign action
        """

        return self.board_size**2+1

    def execute_move(self, action, player):
        """
        Execute a move on pachi py board and return the obtained
        board(assuming copy has been created)
        """

        curr_player = pachi_py.BLACK if player == -1 else pachi_py.WHITE

        if action == self.get_pass_action():
            new_board = self.board.play(pachi_py.PASS_COORD, curr_player)
        elif action == self.get_resign_action():
            new_board = self.board.play(pachi_py.RESIGN_COORD, curr_player)
        else:
            a_x, a_y = action // self.board_size, action % self.board_size
            new_board = self.board.play(
                self.board.ij_to_coord(a_x, a_y), curr_player)

        return GoBoard(self.board_size, new_board)

    def get_legal_moves(self, player):
        """
        Iterate and find out legal position moves
        """

        board_arr = self.board.encoded()

        legal_moves = []

        for pos_x in range(self.board_size):
            for pos_y in range(self.board_size):
                if self.is_legal_action(board_arr, (pos_x, pos_y), player):
                    legal_moves.append(self.board_size*pos_x + pos_y)

        return legal_moves

    def is_legal_action(self, board_arr, action, player):
        """
        Basic checks if the given action on a board is legal
        """

        (pos_x, pos_y) = action

        if not (0 <= pos_x < self.board_size and 0 <= pos_y < self.board_size):
            return False

        if not np.all(board_arr[:, pos_x, pos_y] == EMPTY):
            return False

        # curr_player = BLACK if player == -1 else WHITE
        opp_player = BLACK if player == 1 else WHITE

        if pos_x > 0 and not np.all(board_arr[:, pos_x-1, pos_y] == opp_player):
            return True

        if pos_x != self.board_size-1 and not np.all(board_arr[:, pos_x+1, pos_y] == opp_player):
            return True

        if pos_y > 0 and not np.all(board_arr[:, pos_x, pos_y-1] == opp_player):
            return True
        if pos_y != self.board_size-1 and not np.all(board_arr[:, pos_x, pos_y+1] == opp_player):
            return True

        return False
