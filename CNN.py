# Importing the libraries
import os

# Import the libraries
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

class ImageClassifierNet(nn.Module):
    def __init__(self, n_channels=3):
        super(ImageClassifierNet, self).__init__()
        ######################
        #   YOUR CODE HERE   #
        ######################
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4,4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #self.drop_out = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(7*7*64, 1000)
        #self.fc2 = nn.Linear(1000, 10)
        self.fc = nn.Sequential(nn.Linear(4*7*7, 10))
    def forward(self, X):
        ######################
        #   YOUR CODE HERE   #
        ######################
        out = self.layer1(X)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out