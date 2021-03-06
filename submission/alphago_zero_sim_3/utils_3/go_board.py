"""
Wrapper over Board class of pachiy_py
"""

import numpy as np
import pachi_py

BLACK = np.array([1, 0, 0])
WHITE = np.array([0, 1, 0])
EMPTY = np.array([0, 0, 1])

MAX_MOVES = 250


class GoBoard():
    """
    Wrapper over board class to modify representations and return new copies.
    """

    def __init__(self, board_size, board=None,
                 player=-1, done=False, last_passed=False, history=None, move_num=0):
        self.board_size = board_size

        self.pass_action = board_size**2
        self.resign_action = board_size**2 + 1

        assert player in [-1, 1]

        self.curr_player = player
        self.done = done
        self.last_passed = last_passed
        self.move_num = move_num

        if history is None:
            self.history = [None]*7
        else:
            self.history = history

        if board is None:
            self.board = pachi_py.CreateBoard(board_size)
        else:
            self.board = board

    def set_move_num(self,move_num):
        self.move_num = move_num

    def coord_to_action(self, action_coord):
        """
        Converts Pachi coordinates to actions
        """
        if action_coord == pachi_py.PASS_COORD:
            return self.pass_action
        if action_coord == pachi_py.RESIGN_COORD:
            return self.resign_action
        i, j = self.board.coord_to_ij(action_coord)
        return i*self.board_size + j

    def action_to_coord(self, action):
        """
        Converts actions to Pachi coordinates
        """
        if action == self.pass_action:
            return pachi_py.PASS_COORD
        if action == self.resign_action:
            return pachi_py.RESIGN_COORD
        return self.board.ij_to_coord(action // self.board_size, action % self.board_size)

    def str_to_action(self, string):
        """
        Convert D6 type coordinates to  actions
        """
        return self.coord_to_action(self.board.str_to_coord(string.encode()))

    def execute_move(self, action, player):
        """
        Execute a move on pachi py board and return the obtained
        board(assuming copy has been created)
        """
        # print(player,self.curr_player, flush=True)
        assert not self.done
        assert player == self.curr_player
        assert 0 <= action <= self.board_size**2 + 2

        done = False
        last_passed = False

        curr_player = pachi_py.BLACK if player == -1 else pachi_py.WHITE

        if action == self.pass_action:
            last_passed = True
            if self.last_passed:
                done = True
                # print('2 Passes Done', flush=True)
            new_board = self.board.play(pachi_py.PASS_COORD, curr_player)
        elif action == self.resign_action:
            done = True
            # print('Someone Resigned', flush=True)
            new_board = self.board.play(pachi_py.RESIGN_COORD, curr_player)
        else:
            a_x, a_y = action // self.board_size, action % self.board_size
            new_board = self.board.play(
                self.board.ij_to_coord(a_x, a_y), curr_player)

        if self.move_num == -1:
            move_num = -1
        elif self.move_num == MAX_MOVES:
            done = True
            move_num = self.move_num+1
        else:
            move_num = self.move_num+1

        new_history = [self.board] + self.history[:6]
        # print(len(new_history), flush=True)
        return GoBoard(self.board_size, new_board, -1*player, done, last_passed, new_history, move_num)

    def get_legal_moves_old(self, player):
        """
        Iterate and find out legal position moves
        """

        curr_player = pachi_py.BLACK if player == -1 else pachi_py.WHITE

        board_arr = self.board.encode()
        init_legal_coords = self.board.get_legal_coords(curr_player)

        legal_moves = []
        # print(init_legal_coords)
        for coord in init_legal_coords:
            # print(coord)
            pos_x, pos_y = self.board.coord_to_ij(coord)
            if coord >= 0 and self.is_legal_action_old(board_arr, (pos_x, pos_y), curr_player):
                legal_moves.append(self.board_size*pos_x + pos_y)

        return legal_moves

    def get_legal_moves(self, player):
        """
        Iterate and find out legal position moves
        """
        # return self.get_legal_moves_old(player)

        curr_player = pachi_py.BLACK if player == -1 else pachi_py.WHITE

        init_legal_coords = self.board.get_legal_coords(curr_player)

        legal_moves = []
        for coord in init_legal_coords:
            if coord >= 0 and self.is_legal_action(coord, curr_player):
                pos_x, pos_y = self.board.coord_to_ij(coord)
                legal_moves.append(self.board_size*pos_x + pos_y)

        return legal_moves

    def is_legal_action_old(self, board_arr, action, player):
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

    def is_legal_action(self, action, player):
        """
        Simple is legal action check.
        """
        try:
            # print(action, player)
            _ = self.board.play(action, player)
        except pachi_py.IllegalMove:
            return False

        return True

    def print_board(self):
        """
        Show complete game state.
        """
        color_to_play = 'Black' if self.curr_player == -1 else 'White'
        print('To play: {}\n{}'.format(
            color_to_play, self.board.__repr__().decode()), flush=True)

    def is_terminal(self):
        """
        Check if state is terminal (double pass or resign)
        """
        board_terminal = self.board.is_terminal
        self_check = self.done

        return board_terminal or self_check

    def score(self, komi):
        """
        Score the board configuration
        """
        return self.board.official_score + komi

    def get_numpy_form(self, history, player):
        """
        Convert into ML input form
        """

        history_reps = []

        stones = self.board.encode()

        if player is None or player == -1:
            history_reps.append(stones[:2, :, :])
        else:
            history_reps.append(stones[1:2, :, :])
            history_reps.append(stones[:1, :, :])

        if history:
            for board in self.history:
                if board is None:
                    stones = np.zeros((2, self.board_size, self.board_size))
                else:
                    stones = board.encode()

                if player is None or player == -1:
                    history_reps.append(stones[:2, :, :])
                else:
                    history_reps.append(stones[1:2, :, :])
                    history_reps.append(stones[:1, :, :])

        combined = np.concatenate(history_reps, axis=0)
        combined_in_order = np.transpose(combined, (1, 2, 0))

        return combined_in_order
