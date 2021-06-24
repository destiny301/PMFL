# meta_model
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np


class Model(nn.Module):

    def __init__(self, lib_sz, nclass):

        super(Model, self).__init__()
        self.fc1 = nn.Linear(lib_sz, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc = nn.Linear(16, nclass)
        self.act = nn.Softmax()

    def forward(self, x, length):
        # print('input shape:', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x