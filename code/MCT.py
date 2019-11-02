class MCT():
    def __init__(self, numSimulations, gameBoard, net):
        self.board = gameBoard
        self.numSimulations = numSimulations
        self.neuralNet = net
        self.Nsa = {}
        self.Wsa = {}
        self.Qsa = {}
        self.Psa = {}
        self.terminals = {}

    def simulate(self, boardState, temperature):

        # This step is done in parallel in paper.
        for _ in range(self.numSimulations):
            self.search(boardState)

        if temperature == 0:
            # Return Best Action
            raise NotImplementedError

        state = self.board.getCompressedBoardState(boardState)

        visit_counts = []
        tot_count = 0
        for action in range(self.board.getActionSpaceSize()):
            if (state, action) in self.Nsa:
                visit_counts.append(
                    self.Nsa[(state, action)]**(1.0/temperature))
                tot_count += visit_counts[-1]
            else:
                visit_counts.append(0)

        action_probs = []
        for _, val in enumerate(visit_counts):
            action_probs.append(val/tot_count)

        return action_probs

    def search(self, boardState):
        state = self.board.getCompressedBoardState(boardState)

        if state not in self.terminals:
            self.terminals[state] = self.board.checkTerminal(boardState)

            if self.terminals[state] != 0:
                # Terminal State found.
                return -1 * self.terminals[state]

            # New Node found. Let's expand this.
            self.neuralNet.evaluateBoard(boardState)
