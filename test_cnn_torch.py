""" Just the CNN architecture itself
    in pytorch framework"""

# %%
import torch
import torch.nn as nn
import numpy as np

# %%
RSEED = 66
# Seed for random number generator (makes it reproducible)

# %%
class ThreshNet(nn.Module):
    """
    A deep learning convolutional neural network
    Associates a global map of an environmental variable (such as air temperature)
    with a regional value for time-until-a-threshold is met (such as years until 1.5 C warming)

    Structure:

    """
    def __init__(self):
        """
        Initialise ThreshNet
        
        Args:
            None
            
        Returns:
            Nothing 
        """
        super(ThreshNet, self).__init__()
#        self.pad = nn.CircularPad1d(10)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2D(2)

    def forward(self, x):
        """
        Forward pass of ThreshNet
        """
#        x = self.pad(x)
        x = self.pool1(nn.Functional.relu(self.conv1(x)))
        for i in range(0,2):
            x = self.pool1(nn.Functional.relu(self.conv2(x)))
        z = torch.flatten(x)

        