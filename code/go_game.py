"""
Game board environment manager
"""

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

        validMoveIndicator = [0]*self.getActionSpaceSize()
        legalMoves = board.get_legal_moves(player)

        validMoveIndicator[-1] = 1
        validMoveIndicator[-2] = 1

        for action in legalMoves:
            validMoveIndicator[action] = 1

        return validMoveIndicator

    def getGameEnded(self, board, player):
        """
        Check if game has ended and who has won
        """

        if not board.is_terminal():
            return 0

        # -1 BLACK(1) 1 WHITE(2)
        return self.decide_winner()
        
    def getCanonicalForm(self, board, player):
        """
        ???????????
        """
        # TODO : Need to see what exactly is done here
        return board.get_numpy_form()

    def getSymmetries(self, board, pi):
        """
        Randomly generate symmetries
        """
        # TODO
        return board

    def stringRepresentation(self, board):
        """
        Get representation of state as string
        """
        # TODO
        return board.tostring()

    def getScore(self, board, player):
        """
        Get score according to player
        """
        # Official Score is positive if white is winning
        # White is player 1
        # Black is player -1

        return player * board.score(self.komi)

    def decide_winner(self, board):
        """
        Decide on the winner according to current board.
        """
        score = board.score(self.komi)

        if score == 0:
            return 0

        return 1 if score > 0 else -1
