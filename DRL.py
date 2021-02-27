import torch
import torch.nn as nn


##

class DQNAgent(nn.Module):

    def __init__(self, name, lr=1e-5, gamma=0.99):
        super(DQNAgent, self).__init__()
        self.name = name
        self.in_dim = 128
        self.out_dim = 16
        self.lr = lr
        # self.gamma = gamma

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_dim)
        )

        self.optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.lr)

    def forward(self, state):
        if state is None:
            return 0.
        return self.fc(state)
