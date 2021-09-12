import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.lin1 = nn.Linear(state_size, 32)
        #self.bn1 = nn.BatchNorm2d(32)
        self.lin2 = nn.Linear(32, 128)
        #self.bn2 = nn.BatchNorm2d(128)
        self.lin3 = nn.Linear(128, 32)
        #self.bn3 = nn.BatchNorm2d(32)
        self.out = nn.Linear(32, action_size)
        
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
#         x = F.relu(self.bn1(self.lin1(state)))
#         x = F.relu(self.bn2(self.lin2(x)))
#         x = F.relu(self.bn3(self.lin3(x)))
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.out(x)
        return x
