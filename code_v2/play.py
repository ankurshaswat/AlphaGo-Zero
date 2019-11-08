import numpy as np

from MCT import MCT

BLACK=-1
WHITE=1

def generateEpisodes(nnet, numEpisodes=20, numSimulations=25, numStepsForTempChange=20): #self play
	# numEpisodes=args.numEpisodes
	# numSimulations=args.numSimulations
	# numStepsForTempChange=args.numStepsForTempChange

	train=[] #training data, to be filled with (state, action_prob, value) as encountered during the game

	#create game
	game=Game()
	
	#initial board
	board=game.getInitBoard()

	mct=MCT(nnet, game, numSimulations)

	num_steps=0
	MAX_STEPS=13*13

	player=BLACK #game always starts with BLACK

	temp=1
	for ep in range(numEpisodes):
		episode=[]
		while True:
			num_steps+=1
			if(num_steps>=numStepsForTempChange):
				temp=0

			#get action probabilities
			action_prob=mct.actionProb(board, player, temp)

			#get valid actions
			valids=game.getValidMoves(board,player)

			#append board(randomly flipped/rotated),action_prob to training data (value/reward will be added later)
	        l = game.getSymmetries(board, action_prob, valids, player, history=False)
	        for b,p,v in l:
	        	episode.append([b, p, v, 0]) #dummy reward for now

	        #pick action and play
	        next_action =  np.random.choice(len(action_prob), p=action_prob)

	        board=game.getNextState(board, player, next_action)

	        #update player
	        player=-player

	        reward=None

	        if(num_steps>=MAX_STEPS) or board.is_terminal(): #maximum steps reached
	            reward = game.decide_winner(board, player)
	        

	        if(reward is not None):
	        	for i in reversed(range(len(train))):
		        	reward=-reward #flip reward, as player alternates
	        		episode[i][-1]=reward
	        	break

	    train+=episode

	return train

