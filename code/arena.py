import random 
from go_game import GoGame

if __name__ == "__main__":
    game = GoGame(13,5.5)

    board = game.getInitBoard()
    player = -1

    for i in range(1000):
        actions = game.getValidMoves(board,player)
        # print(actions)
        selected_action = None
        possible_actions = []
        for action , indicator in enumerate(actions):
            if indicator == 1:
                # selected_action = action
                # break
                possible_actions.append(action)

        selected_action = random.choice(possible_actions)        
        # print(selected_action)
        # print(board)
        # print(board.board.get_legal_coords(1))
        # print(board.board.get_legal_coords(2))
        (board,player) = game.getNextState(board,player,selected_action)
        print(game.get_numpy_rep(board).shape)