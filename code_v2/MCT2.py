import math

import numpy as np

INF=10000000
NOISE_WEIGHT = 0.25

EPS = 1e-8




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

        # self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        # self.Nsa = {}       # stores #times edge s,a was visited
        # self.Ns = {}        # stores #times board s was visited
        # self.Ps = {}        # stores initial policy (returned by neural net)




        self.edge_visit_cnt={}
        self.state_visit_cnt={}
        self.Q={}
        self.policy={}

        # self.Es = {}        # stores game.getGameEnded ended for board s
        #self.Vs = {}        # stores game.getValidMoves for board s

        self.numSimulations = args.numSimulations
        self.args = args

        self.greedy=greedy
        self.noise=noise

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
            self.search(board, player)


            # except pachi_py.IllegalMove:
            # print('Illegal action taken. Continuing.')
            # continue

        state = self.game.get_string_rep(board, player, history=False)

        cnt=[]


        for action in range(self.game.get_action_space_size()):
            tup=(state, action)
            if(tup in self.edge_visit_cnt):
                cnt.append(self.edge_visit_cnt[tup])
            else:
                cnt.append(0)

        
        if temp == 0: #deterministic
            best = np.argmax(cnt)
            probs = [0]*len(cnt)
            probs[best] = 1
            return probs

        cnt_with_temp=[]
        for i in cnt:
            cnt_with_temp.append(i**(1.0/temp))
        # cnt = [x**(1./temp) for x in cnt]
        # print(counts, flush=True)
        probs = [x/float(sum(cnt)) for x in cnt] #normalize

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
        return probs

    def search(self, board, player):
        # print(player)
        #---string representation of board--#
        # player independent repr, without any history
        state = self.game.get_string_rep(board, player, history=False)
        #-----------------------------------#

        game_end_result = self.game.get_game_ended(board, player)

        if game_end_result != 0:
            # terminal node
            return -game_end_result

        if state not in self.policy: #self.state_visit_cnt: #leaf
            #---get board representation before feeding to neuralNet---#
            # pass player here, another option is to concat player to board repr
            board_repr = self.game.get_numpy_rep(board, player, history=False)
            if(not self.greedy):
                self.policy[state], value = self.nnet.predict(board_repr)
            else:
                self.policy[state], value = self.greedy_action(board, player)
        
            valids = self.game.get_valid_moves(board, player)
            self.policy[state] = self.policy[state]*valids      # masking invalid moves
            
            if  np.sum(self.policy[state]) == 0:
                self.policy[state]=self.policy[state]+ valids
                
            self.policy[state] /= np.sum(self.policy[state])


            #self.Vs[s] = valids
            self.state_visit_cnt[state] = 0
            return -value

        #valids = self.Vs[s]
        valids = self.game.get_valid_moves(board, player)
        
        best_action, best_bound= self.game.get_action_space_size()-1, -INF

        # pick action with best upper confidence bound
        for action in np.random.permutation(list(range(self.game.get_action_space_size()))): #some randomness in action choice
        # for a in range(self.game.get_action_space_size()):

            tup=(state,action)
            if valids[action]:
                if tup in self.Q:
                    u = self.Q[tup] + self.args.cpuct*self.policy[state][action] * math.sqrt(self.state_visit_cnt[state])/(1+self.edge_visit_cnt[tup])
                else:
                    u = self.args.cpuct * self.policy[state][action]*math.sqrt(self.state_visit_cnt[state] + EPS)    

                if u > best_bound:
                    best_bound = u
                    best_action = action

    
        # next_s, next_player = self.game.getNextState(board, player, a)
        next_s = self.game.get_next_state(board, player, best_action)

        next_player = -player
        # next_s = self.game.getCanonicalForm(next_s, next_player)

        value = self.search(next_s, next_player)

        tup=(state,action)
        if tup in self.Q:
            self.edge_visit_cnt[tup] += 1

            self.Q[tup] = (self.edge_visit_cnt[tup] *self.Q[tup] + value)/(self.edge_visit_cnt[tup]+1)

        else:
            self.edge_visit_cnt[tup] = 1

            self.Q[tup] = value

        self.state_visit_cnt[state] += 1
        return -value

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


