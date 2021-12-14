# =====Tianyi Zhang======
# read saved numpy data and split

# input: drug name:...
#        admission diagnosis:...
# predict: disease(apacheadmissiondx)[CVA, CHF, Sepsis, Asthma, pulmonary, Pneumonia, Cardiomyopathy]
#        unitdischargeoffset, unittype, unitstaytype, unitdischargelocation, unitdischargestatus

import os
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
import random
from sklearn.utils import shuffle
import re, io
import torch
from torch.nn.utils.rnn import pad_sequence

class EICUReader():

    def __init__(self, root):
        # read data
        self.datasilospath = os.path.join(root, '/data/datasilo.npz')
        datapath = os.path.join(root, '/data/diagdata.npz')

        dataset = np.load(datapath, allow_pickle=True)
        self.tensor = dataset['data'].astype(np.int64)
        self.text2index = dataset['t2i'].item()
        self.index2text = dataset['i2t'].item()
        self.vocab_sz = dataset['v_sz']
        print("data reading finished!")
        print("vocab_sz:", self.vocab_sz)

        # read label
        self.label = np.load(os.path.join(root, '/data/disease.npz'), allow_pickle=True)['arr_0'].astype(np.int64)


        print("data reading finished!")
        self.tag = [-1]*self.tensor.shape[0] # make sure no repitition for all silos
        self.tensor, self.label = shuffle(self.tensor, self.label)
        print("Total data and Label shape:", self.tensor.shape, self.label.shape)

    # return the vocab size
    def get_lib_sz(self):
        return self.vocab_sz

    # return the vector size for each input text medical record
    def get_text_length(self):
        return self.tensor.shape[1]

    # for testing, return all data and the testing label
    def testdata(self):
        return self.tensor, self.label[:, 6]

    # split all data into 5 different silos
    def split2save(self):
        x_silos = []
        y_silos = []
        # classes = [0, 1, 2, 3, 4, 5]
        # selected_cls = np.random.choice(classes, n_silo, False) # also could select any number of silos randomly
        selected_cls = [0, 1, 5, 4, 2] # use this order to split more fairly
        # x_cls = self.label[:, 0]
        # print(np.count_nonzero(x_cls == 0), np.count_nonzero(x_cls == 1))
        for cur_class in selected_cls:
            x_silo = []
            y_silo = []
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 0 and self.tag[index] == -1:
                    x_silo.append(self.tensor[index])
                    y_silo.append(0)
                    self.tag[index] = cur_class
                if len(y_silo) > 3000:
                    break

            class_sz = len(y_silo)*2
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 1 and self.tag[index] == -1:
                    x_silo.append(self.tensor[index])
                    y_silo.append(1)
                    self.tag[index] = cur_class
                if len(y_silo) == class_sz:
                    break

            x_silo, y_silo = shuffle(x_silo, y_silo)
            x_silo = np.array(x_silo).astype(np.int64)
            y_silo = np.array(y_silo).astype(np.int64)
            x_silos.append(x_silo)
            y_silos.append(y_silo)
            print('num of classes:\t', len(set(y_silo)))
            print("Training class {} size: {}".format(cur_class, len(x_silo)))
        x_silos = np.array(x_silos, dtype=object)
        y_silos = np.array(y_silos, dtype=object)
        np.savez(self.datasilospath, x=x_silos, y=y_silos)
        print(x_silos.shape, x_silos.dtype)
        print(y_silos.shape, y_silos.dtype)

    # simply select training silos from saved silos
    def getTrain(self, disease, n_silo):
        # make sure the test silo is different from all training silos
        selected_cls = [3, 1, 2]
        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("train data and label reading finished!")
        else:
            print("train set not exist!")
        return x_silos, y_silos

    def getTest(self, disease):
        selected_cls = 0
        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("test data and label reading finished!")
        else:
            print("test set not exist!")

        return x_silos, y_silos