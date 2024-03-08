import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FF_dataset(Dataset):

    """
    custom dataset class for training the feedforward network
    Parameters:
        data: N x K array of training data. (N = number of subjects, K = number of features)
        targets: N x 1 array of binary classifications
    """

    def __init__(self, data, targets):
        self.data    = torch.Tensor(data)
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


class FF_network(nn.Module):

    """
    FF network model. Model is a feedforward network composed of ReLU units.
    Model is trained with dropout. No other regularization

    Parameteters:
        input_dim: number of input units
        num_hidden_layers: number of layers between input and output layers
        num_units_per_layer: self explanatory
        dropout_p: fraction of units within a given layer that are 'turned off' for one batch.
        This layer is inactivated (i.e., there is no dropout) during model evaluation.
    """

    def __init__(self, input_dim, num_hidden_layers, num_units_per_layer, dropout_p):
        super().__init__()

        # define the nonlinearity (one for the hidden layers, one for the output)
        self.nonLinearity = nn.ReLU()
        self.outputNonlinearity = nn.Sigmoid()

        # define the input layer
        self.fc1 = nn.Linear(input_dim, num_units_per_layer)

        # define the hidden layers
        self.fc2 = nn.ModuleList()
        for ii in range(num_hidden_layers):
            self.fc2.append(nn.Linear(num_units_per_layer, num_units_per_layer))

        # define the output layer
        self.fc3 = nn.Linear(num_units_per_layer, 1)

        # define our dropout layer that will sit before each hidden layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):

        # input layer
        out = self.nonLinearity(self.fc1(x))

        # hidden layers
        for fc_hidden in self.fc2:
            out = self.dropout(out)
            out = self.nonLinearity(fc_hidden(out))

        # output layer
        out = self.fc3(out)
        out = self.outputNonlinearity(out)

        return out










