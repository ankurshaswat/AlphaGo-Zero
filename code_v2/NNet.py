"""
File to manage everything in neural net.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNet(nn.Module):
    """
    Class managing everthing in neural net.
    """
    def __init__(self, game, args):
        self.board_size, _ = game.getBoardSize()
        self.action_size = game.getActionSpaceSize()
        self.args = args

        super(NNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * args.history, args.num_channels, 3, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, 3, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, 3),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, 3),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(args.num_channels*(self.board_size-4)
                      * (self.board_size-4), 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=self.args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=self.args.dropout)
        )

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 2, self.board_size, self.board_size)
        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)
        s = self.conv4(s)

        s = s.view(-1, self.args.num_channels *
                   (self.board_size-4)*(self.board_size-4))

        s = self.fc1(s)
        s = self.fc2(s)

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def start_training(self, examples, args):
        optimizer = optim.Adam(self.parameters(), lr=0.001, momentum=0.9,weight_decay=args.l2_regularization)

        mse_loss = nn.MSELoss()
        crossEntr_loss = nn.CrossEntropyLoss()
        
        for epoch in range(args.epochs):
            self.train()

            all_ids = list(range(len(examples)))
            random.shuffle(all_ids)

            i = 0
            batch_num = 0
            running_loss_v = 0
            running_loss_pi = 0

            while(i<len(examples)):
                selected_examples = all_ids[i:i+args.batch_size]

                example_batch, pi, v = list(
                    zip(*[examples[i] for i in selected_examples]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pi = torch.FloatTensor(np.array(pis))
                target_v = torch.FloatTensor(np.array(vs).astype(np.float64))

                predicted_pi, predicted_v = self(example_batch)

                loss_pi = crossEntr_loss(predicted_pi,target_pi)
                loss_v = mse_loss(predicted_v,target_v)

                running_loss_pi += loss_pi.item()
                running_loss_v += loss_v.item
                total_loss = loss_pi + loss_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                batch_num += 1
                i += args.batch_size

    def save_checkpoint(self, path):
        torch.save({'state_dict': self.state_dict()}, path)

    def load_checkpoint(self, path):
        location = None if CUDA else 'cpu'
        saved_model = torch.load(path, map_location=location)
        self.load_state_dict(saved_model['state_dict'])

    def predict(self, numpy_board):
        board = torch.FloatTensor(numpy_board)
        if self.args.cuda:
            board = board.contiguous().cuda()

        board = board.view(2, self.board_size, self.board_size)
        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
