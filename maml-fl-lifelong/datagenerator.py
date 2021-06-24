# =====Destiny======
# generate data

import os
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.utils import shuffle
import re, io
import torch
from torch.nn.utils.rnn import pad_sequence

class I2B2Dataset(Dataset):

    def __init__(self, x, y, ratio, mode = 'train'):
        if mode == 'test':
            self.tensor = np.array(x).astype(np.int64)
            self.label = np.array(y).astype(np.int64)
            self.xtr = self.tensor[:int(self.tensor.shape[0]*ratio),:]
            self.ytr = self.label[:int(self.label.shape[0]*ratio)]
            self.xte = self.tensor[int(self.tensor.shape[0]*ratio):,:] # memory Ath-->0.95
            self.yte = self.label[int(self.label.shape[0]*ratio):]
            # self.xtr = self.tensor[:500, :]
            # self.ytr = self.label[:500]
            # self.xtr, self.ytr = shuffle(self.xtr, self.ytr)
            # print(self.xtr.shape, self.ytr.shape)

            # self.xte = self.tensor[1000:, :]
            # self.yte = self.label[1000:]
            # self.xte, self.yte = shuffle(self.xte, self.yte)
            # print(self.xte.shape, self.yte.shape)
            print("Test ds training shape:", self.xtr.shape, self.ytr.shape)
        else:
            self.tensor = x
            self.label = y
    
    def createDataset(self, n_silo, n_batch):
        x = []
        y = []

        for i in range(n_batch):
            xbatch = []
            ybatch = []
            for j in range(n_silo):
                batchsz = self.tensor[j].shape[0]//n_batch
                if i == 0:
                    print('batch size of silo {}:'.format(j), batchsz)
                xbatch.append(self.tensor[j][i*batchsz:(i+1)*batchsz, :])
                ybatch.append(self.label[j][i*batchsz:(i+1)*batchsz])
            x.append(xbatch)
            y.append(ybatch)

        return x, y

    def __len__(self):
        return len(self.xtr)

    def __getitem__(self, index):
        return self.xtr[index], self.ytr[index]

    def get_testdata(self):
        return self.xte, self.yte