import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
CUDA = True


class NNet(nn.Module):
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSpaceSize()
        self.args = args

        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(2, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # batch_size x 1 x board_x x board_y
        s = s.view(-1, 2, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels *
                   (self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 512

        # batch_size x action_size
        pi = self.fc3(s)
        # batch_size x 1
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def start_training(self, examples, args):
        optimizer = optim.Adam(lr= ????, momentum= ???)

        for epoch in range(args.epochs):
            self.train()

            while batch_idx < int(len(examples)/args.batch_size):
                selected_examples = np.random.randint()
                example_batch, pi, v = list(
                    zip(*[examples[i] for i in selected_examples]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pi = torch.FloatTensor(np.array(pis))
                target_v = torch.FloatTensor(np.array(vs).astype(np.float64))

                predicted_pi, predicted_v = self(example_batch)
                loss_pi = -torch.sum(target_pi*predicted_pi) / \
                    target_pi.size()[0]
                loss_v = torch.sum((target_v-predicted_v.view(-1))
                                   ** 2)/target_v.size()[0]

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                batch_idx += 1

    def save_checkpoint(self, path):
        torch.save({'state_dict': self.state_dict()}, path)

    def load_checkpoint(self, path):
        location = None if CUDA else 'cpu'
        saved_model = torch.load(path, map_location=location)
        self.load_state_dict(saved_model['state_dict'])

    def predict(self, numpy_board):
        board = torch.FloatTensor(numpy_board)
        if CUDA:
            board = board.contiguous().cuda()

        board = board.view(2, self.board_size, self.board_size)
        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
