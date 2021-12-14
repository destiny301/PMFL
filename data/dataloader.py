# =====Tianyi Zhang======
# generate data batches

import os
from torch.utils.data import Dataset
import numpy as np
# from sklearn.utils import shuffle

class batchloader(Dataset):

    def __init__(self, x, y, ratio, mode = 'train'):
        if mode == 'test':
            self.tensor = np.array(x).astype(np.int64)
            self.label = np.array(y).astype(np.int64)
            self.xtr = self.tensor[:int(self.tensor.shape[0]*ratio),:]
            self.ytr = self.label[:int(self.label.shape[0]*ratio)]
            self.xte = self.tensor[int(self.tensor.shape[0]*ratio):,:] # for Ath-->0.95
            self.yte = self.label[int(self.label.shape[0]*ratio):]

            print("Test ds training shape:", self.xtr.shape, self.ytr.shape)
        else:
            self.tensor = x
            self.label = y

    # create batches for training silos, keep the number of batch for each silo the same
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