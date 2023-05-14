import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        self.max_pooling = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6912, 128)
        self.fc2 = nn.Linear(128, 10)
        return
    
    def forward(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.max_pooling(h)
        h = h.reshape(-1, 6*6*6*32)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        
        return h