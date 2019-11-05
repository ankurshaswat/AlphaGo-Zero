"""
Game board environment manager
"""

import numpy as np

from go_board import GoBoard as Board


class GoGame:
    """
    Game class to use board and perform required operations for training.
    """

    def __init__(self, board_size, komi):
        self.board_size = board_size
        self.komi = komi

    def getInitBoard(self):
        """
        Get a fresh board.
        """

        board = Board(self.board_size)
        return board

    def getBoardSize(self):
        """
        Return board size as tuple
        """

        return (self.board_size, self.board_size)

    def getActionSpaceSize(self):
        """
        Get number of possible actions
        """

        return self.board_size ** 2 + 2

    def getNextState(self, board, player, action):
        """
        Take action and get new board
        """

        new_board = board.execute_move(action, player)
        return (new_board, -player)

    def getValidMoves(self, board, player):
        """
        Get valid moves over whole board
        """

        valid_move_indicator = [0]*self.getActionSpaceSize()
        legal_moves = board.get_legal_moves(player)

        valid_move_indicator[-1] = 1
        valid_move_indicator[-2] = 1

        for action in legal_moves:
            valid_move_indicator[action] = 1

        return np.asarray(valid_move_indicator)

    def getGameEnded(self, board, player):
        """
        Check if game has ended and who has won (in view of player)
        """

        if not board.is_terminal():
            return 0

        # -1 BLACK(1) 1 WHITE(2)
        # 1 if player has won
        # -1 if player has lost
        return self.decide_winner(board, player)

    def get_numpy_rep(self, board, player=None, history=True):
        """
        Get numpy representation of board.
        """
        return board.get_numpy_form(player, history)

    def getSymmetries(self, numpy_board, pi_, rot_num, flip):
        """
        Randomly generate symmetries
        """
        assert rot_num in [0, 1, 2, 3]
        assert len(pi_) == self.board_size**2 + 2
        pi_2d = np.reshape(pi_[:-2], (self.board_size, self.board_size))

        if rot_num > 0:
            new_board = np.rot90(numpy_board, rot_num)
            new_pi = np.rot90(pi_2d, rot_num)
        else:
            new_board = numpy_board
            new_pi = pi_2d

        if flip:
            new_board = np.fliplr(new_board)
            new_pi = np.fliplr(new_pi)

        return new_board, list(new_pi.ravel()) + pi_[-2:]

    def stringRepresentation(self, board, player=None, history=True):
        """
        Get representation of state as string.
        Only concat of board_size*board_size*16 array is returned.
        """
        np_arr = self.get_numpy_rep(board, player, history)
        return self.convert_np_to_string(np_arr)

    def convert_np_to_string(self, np_arr):
        """
        Convert an already generated numpy_representation to string.
        """
        return "".join(str(x) for x in np_arr.ravel())

    def getScore(self, board, player):
        """
        Get score according to player
        """
        # Official Score is positive if white is winning
        # White is player 1
        # Black is player -1

        return player * board.score(self.komi)

    def decide_winner(self, board, player):
        """
        Decide on the winner according to current board.
        1 if player has won
        -1 if player has lost
        """
        score = board.score(self.komi)

        if score == 0:
            return 0

        return player if score > 0 else -player
