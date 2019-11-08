"""
File to manage training and predicting fom neural net.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import NNet1, NNet2, NNet3


class NetTrainer():
    """
    Class to train neural net.
    """

    def __init__(self, game, args):
        self.args = args
        self.board_size = game.getBoardSize()

        if self.args.type == 1:
            self.net = NNet1(game, args)
        elif self.args.type == 2:
            self.net = NNet2(game, args)
        elif self.args.type == 3:
            self.net = NNet3(game, args)
        else:
            print('Specify neural net type correctly.')

    def train(self, examples):
        """
        Train neural net
        """
        optimizer = optim.RMSprop(self.net.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum, weight_decay=self.args.l2_regularization)

        mse_loss = nn.MSELoss()
        entr_loss = nn.CrossEntropyLoss()

        for epoch in range(self.args.epochs):
            self.net.train()

            all_ids = list(range(len(examples)))
            random.shuffle(all_ids)

            i = 0
            batch_num = 0
            running_loss_v = 0
            running_loss_pi = 0

            while i < len(examples):
                selected_examples = all_ids[i:i+self.args.batch_size]

                example_batch, pis, vals = list(
                    zip(*[examples[i] for i in selected_examples]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pi = torch.FloatTensor(np.array(pis))
                target_v = torch.FloatTensor(np.array(vals).astype(np.float64))

                predicted_pi, predicted_v = self.net(example_batch)

                loss_pi = entr_loss(predicted_pi, target_pi)
                loss_v = mse_loss(predicted_v, target_v)

                running_loss_pi += loss_pi.item()
                running_loss_v += loss_v.item
                total_loss = loss_pi + loss_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_num += 1
                i += self.args.batch_size

            print('Epoch {}: Loss_pi {} Loss_v {}'.format(
                epoch, running_loss_pi/batch_num, running_loss_v/batch_num))

    def save_checkpoint(self, path):
        """
        Save Neural Net params.
        """
        torch.save({'state_dict': self.net.state_dict()}, path)

    def load_checkpoint(self, path):
        """
        Load Neural Net Params
        """
        location = None if self.args.cuda else 'cpu'
        saved_model = torch.load(path, map_location=location)
        self.net.load_state_dict(saved_model['state_dict'])

    def predict(self, numpy_board):
        """
        Predict vals and pis.
        """
        board = torch.FloatTensor(numpy_board)
        if self.args.cuda:
            board = board.contiguous().cuda()

        board = board.view(2, self.board_size, self.board_size)
        self.net.eval()
        with torch.no_grad():
            pis, vals = self.net(board)

        return pis.data.cpu().numpy()[0], vals.data.cpu().numpy()[0]
