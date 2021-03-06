import math

import numpy as np

EPS = 1e-8

NOISE_WEIGHT = 0.25


def add_noise(probs):
    probs = np.array(probs)
    probs = (1-NOISE_WEIGHT)*probs + NOISE_WEIGHT * \
        np.random.dirichlet([0.03]*len(probs), 1)
    probs = probs.squeeze()
    return probs
    # return [i for i in probs]





class MCT(object):
    def __init__(self, nnet, game, args, greedy=False, noise=True):
        self.nnet = nnet
        self.game = game

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        # self.Es = {}        # stores game.getGameEnded ended for board s
        #self.Vs = {}        # stores game.getValidMoves for board s

        self.numSimulations = args.numSimulations
        self.args = args

        self.greedy=greedy
        self.noise=noise
        # self.q_sa={}
        # self.n_sa={}
        # self.n_s={}
        # self.valid_s={}

    def actionProb(self, board, player, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
                probs: a policy vector where the probability of the ith action is
                proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.numSimulations):
            # try:

            #set board.move_number to 0
            self.search(board, player)


            # except pachi_py.IllegalMove:
            # print('Illegal action taken. Continuing.')
            # continue

        s = self.game.get_string_rep(board, player, history=False)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.get_action_space_size())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        # print(counts, flush=True)
        probs = [x/float(sum(counts)) for x in counts]

        if(self.noise):
            probs = add_noise(probs)

        valids = self.game.get_valid_moves(board, player)
        #valids = self.Vs[s]
        probs = probs*valids

        probs /= np.sum(probs)

        print(probs[probs>0][:5])
        xx=input()


        probs = probs.tolist()

        # print("DEBUG: {} {}".format(type(probs),len(probs)),flush=True)

        '''
        #add dirichlet noise

        probs=np.array(probs)
        print("probs.shape:",probs.shape, flush=True)
        print(probs[:3])
        probs= (1-NOISE_WEIGHT)*probs+ NOISE_WEIGHT* np.random.dirichlet([0.03]*len(probs),1)
        probs=probs.squeeze()
        print("probs.shape:",probs.shape, flush=True)
        print(probs[:3])
        
        # xx=input()
        '''
        return probs

    def search(self, board, player):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
                v: the negative of the value of the current canonicalBoard
        """
        # print(player)
        #---string representation of board--#
        # player independent repr, without any history
        s = self.game.get_string_rep(board, player, history=False)
        #-----------------------------------#

        # if s not in self.Es:
        # self.Es[s] = self.game.get_game_ended(board, player)
        game_end_result = self.game.get_game_ended(board, player)
        # print("Winner:",self.Es[s], flush=True)

        if game_end_result != 0:
            # terminal node
            return -game_end_result

        if s not in self.Ps:
            # leaf node

            #---get board representation before feeding to neuralNet---#
            # pass player here as well, another option is to concat player to board repr
            board_repr = self.game.get_numpy_rep(board, player, history=False)
            if(not self.greedy):
                self.Ps[s], v = self.nnet.predict(board_repr)
            else:
                self.Ps[s], v = self.greedy_action(board, player)
            #------------------------------#

            valids = self.game.get_valid_moves(board, player)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # TODO : print("All valid moves were masked, do workaround.", flush=True)
                print("All valid moves were masked, do workaround.",
                      valids, flush=True)
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            #self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        #valids = self.Vs[s]
        valids = self.game.get_valid_moves(board, player)
        cur_best = -float('inf')
        best_act = self.game.get_action_space_size()-1  # -1

        # pick the action with the highest upper confidence bound
        # for a in range(self.game.get_action_space_size()):
        for a in np.random.permutation(list(range(self.game.get_action_space_size()))):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct*self.Ps[s][a] * \
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * \
                        self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # next_s, next_player = self.game.getNextState(board, player, a)
        next_s = self.game.get_next_state(board, player, a)

        next_player = -player
        # next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, next_player)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)]+1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def greedy_action(self, board, player):

        valids = self.game.get_valid_moves(board, player)



        num_actions=len(valids)
        high_score = 0
        greedy_action = None

        for action, is_valid in enumerate(valids):
            if not is_valid:
                continue
            new_board = self.game.get_next_state(board, player, action)
            score = self.game.get_score(new_board, player) 
            
            if score >= high_score: # Modify if multiple high scores
                greedy_action = action # The greedy action

        policy=[0.0]*num_actions
        policy[action]=1.0

        policy=np.array(policy)
        value= self.game.get_score(board, player)


        return policy, value

    # def actionProb(self, board, player, temp=1):
    # 	pass

    # def search(self, board, player):
    # 	pass
