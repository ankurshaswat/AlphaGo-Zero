class Trainer:
    def __init__(self, game_board, net):
        self.net = net
        self.game_board = game_board

    def generate_epi(self):
        board = self.game_board.reset()
        player = False  # 0
        epi = 0

        while True:
            epi += 1
            board_player_view = self.game_board.get_view(player)

            action_probs = self.mcts_tree.search_next(board_player_view)
