"""
All models to test.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNet1(nn.Module):
    """
    Neural Net module 1.
    """

    def __init__(self, game, args):
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSpaceSize()
        self.args = args

        super(NNet1, self).__init__()

        num_init_channels = 2 if not args.history else 16

        self.conv1 = nn.Sequenti0al(
            nn.Conv2d(num_init_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128*(self.board_size-4) * (self.board_size-4), 1024),
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

        pi_probab_dist = self.fc3(s)
        val = self.fc4(s)

        return F.log_softmax(pi_probab_dist), torch.tanh(val)


class NNet2(nn.Module):
    """
    Neural Net module 2.
    """

    def __init__(self, game, args):
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSpaceSize()
        self.args = args

        super(NNet2, self).__init__()

        num_init_channels = 2 if not args.history else 16

        self.conv1 = nn.Sequenti0al(
            nn.Conv2d(num_init_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128*(self.board_size-2) * (self.board_size-2), 512),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=self.args.dropout)
        )

        self.fc2 = nn.Linear(512, self.action_size)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 2, self.board_size, self.board_size)
        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)

        s = s.view(-1, self.args.num_channels *
                   (self.board_size-2)*(self.board_size-2))

        s = self.fc1(s)

        pi_probab_dist = self.fc2(s)
        val = self.fc3(s)

        return F.log_softmax(pi_probab_dist), torch.tanh(val)


class NNet3(nn.Module):
    """
    Neural Net module 3 (ResNet).
    """

    def __init__(self, game, args):
        self.board_size = game.get_board_size()
        self.action_size = game.get_action_space_size()
        self.args = args

        super(NNet3, self).__init__()

        num_init_channels = 2 if not args.history else 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_init_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128*(self.board_size-2) * (self.board_size-2), 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=self.args.dropout)
        )

        self.fc2 = nn.Linear(512, self.action_size)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 2, self.board_size, self.board_size)
        s = self.conv1(s)
        s_saved = s.clone()
        s = self.conv2(s)
        s += s_saved
        s = self.conv3(s)

        s = s.view(-1, 128 * (self.board_size-2)*(self.board_size-2))

        s = self.fc1(s)

        pi_probab_dist = self.fc2(s)
        val = self.fc3(s)

        return F.log_softmax(pi_probab_dist, dim=1), torch.tanh(val)
