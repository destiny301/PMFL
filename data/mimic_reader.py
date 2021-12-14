# =====Tianyi Zhang======
# read MIMIC-CXR dataset and split into silos

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re, io
import torch
from torch.nn.utils.rnn import pad_sequence

class mimicReader():

    def __init__(self, root):
        datafolder = os.path.join(root, 'files')
        labelpath = os.path.join(root, 'mimic-cxr-2.0.0-chexpert.csv')
        datasetpath = os.path.join(root, 'data/mimicDataset.npz') # the whole dataset as numpy array
        self.datasilospath = os.path.join(root, 'data/datasilo.npz') # all silos as numpy array

        self.data = []
        self.label = []
        self.tensor = []
        self.text2index = {}
        self.index2text = {}
        self.vocab_sz = 0

        # if original files have been read and saved, then directly load
        if os.path.exists(datasetpath):
            dataset = np.load(datasetpath, allow_pickle=True)
            self.tensor = dataset['tensor']
            self.label = dataset['label']
            self.text2index = dataset['text2index'].item()
            self.index2text = dataset['index2text'].item()
            self.vocab_sz = dataset['vocab_sz']
            print("data and label reading finished!")
            print("vocab_sz:", self.vocab_sz)
        # if not, then read all files
        else:
            print("Dataset not existed, need to read!")
            # read mimic-cxr labels
            df = pd.read_csv(labelpath, usecols=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 
                        'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax'])
            df.fillna(0, inplace=True) # set None to be zero
            array = df.to_numpy()
            self.label = np.where(array==-1, 0, array).astype(np.int64) # to do......................................................................
            print("labels reading finished!")

            # read mimic-cxr text medical records
            for dir1 in sorted(os.listdir(datafolder)):
                dir1path = os.path.join(datafolder, dir1)
                for dir2 in sorted(os.listdir(dir1path)):
                    dir2path = os.path.join(dir1path, dir2)
                    for files in sorted(os.listdir(dir2path)):
                        filepath = os.path.join(dir2path, files)
                        lines = io.open(filepath, encoding='UTF-8').read().strip().split('\n')
                        self.data.append(self.preprocess(' '.join(lines)))
            print("data reading finished!")
            # print(len(self.data))
            # print(self.data[0])
            self.tokenizer(self.data)

            # save dataset as a numpy array
            np.savez(datasetpath, tensor=self.tensor, label=self.label, 
                        text2index = self.text2index, index2text = self.index2text, vocab_sz = self.vocab_sz)
            print("data saving finished!")

        self.tag = [-1]*self.tensor.shape[0] # make sure no repitition for all silos
        self.tensor, self.label = shuffle(self.tensor, self.label)
        print("Total data and Label shape:", self.tensor.shape, self.label.shape)

    # preprocess text medical records, eg. remove special char
    def preprocess(self, s):

        s = re.sub(r"([?!.,¿])", r" \1", s)
        s = re.sub(r'[" "]+', " ", s)

        s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
        
        s = s.strip()
        return s

    # tokenizer according to space--' ', and transfer to the according index vectors
    def tokenizer(self, data):
        lib = []
        for i in range(len(data)):
            text = data[i].split()
            for j in range(len(text)):
                if text[j] not in lib:
                    lib.append(text[j])

        self.text2index = {m:n for n, m in enumerate(lib)}
        self.index2text = {n:m for n, m in enumerate(lib)}
        self.vocab_sz = len(lib)
        for i in range(len(data)):
            text = data[i].split()
            id = torch.zeros(len(text), dtype = torch.int)
            for j in range(len(text)):
                id[j] = self.text2index[text[j]]
            self.tensor.append(id)
        self.tensor = pad_sequence(self.tensor, batch_first = True, padding_value = 0)
        self.tensor = np.asarray(self.tensor)

    # return the vocab size
    def get_lib_sz(self):
        return self.vocab_sz

    # return the vector size for each input text medical record
    def get_text_length(self):
        return self.tensor.shape[1]
   
    # split all data into 8 different silos, select 8 labels related to lung
    def split2save(self):
        x_silos = []
        y_silos = []
        # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # selected_cls = np.random.choice(classes, n_silo, False) # also could select any number of silos randomly
        selected_cls = [9, 6, 11, 2, 10, 0, 8, 7] # here, directly select 8 labels related to lung, use this order to split more fairly
        for cur_class in selected_cls:
            x_silo = []
            y_silo = []
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 1 and self.tag[index] == -1:
                    x_silo.append(self.tensor[index])
                    y_silo.append(1)
                    self.tag[index] = cur_class # distribute this record/patient into current client/class
            class_sz = len(y_silo)*2 # make sure the number of negative = the number of positive
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 0 and self.tag[index] == -1:
                    x_silo.append(self.tensor[index])
                    y_silo.append(0)
                    self.tag[index] = cur_class
                if len(y_silo) == class_sz:
                    break

            x_silo, y_silo = shuffle(x_silo, y_silo)
            x_silo = np.array(x_silo).astype(np.int64)
            y_silo = np.array(y_silo).astype(np.int64)
            x_silos.append(x_silo)
            y_silos.append(y_silo)
            print("Training class {} size: {}".format(cur_class, len(x_silo)))
        x_silos = np.array(x_silos, dtype=object)
        y_silos = np.array(y_silos, dtype=object)
        np.savez(self.datasilospath, x=x_silos, y=y_silos)
        print('num of silos:', x_silos.shape, x_silos.dtype)

    # simply select training silos from saved silos
    def getTrain(self, disease, n_silo=5):
        # make sure the test silo is different from all training silos
        dd = {'Atelectasis':[1, 2, 3, 4, 7], 
            'Consolidation':[6, 0, 7, 2, 5],
            'LungLesion':[6, 0, 7, 2, 5],
            'LungOpacity':[6, 0, 1, 3, 5],
            'PleuralEffusion':[1, 2, 3, 4, 7],
            'PleuralOther':[1, 2, 3, 4, 7],
            'Pneumonia':[6, 0, 7, 2, 5],
            'Pneumothorax':[6, 0, 1, 3, 5]}

        selected_cls = np.random.choice(dd[disease], n_silo, False) # could set n_silo to be 3, 4, 5, to train with different number of clients

        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("train data and label reading finished!")
        else:
            print("train set not exist!")
        return x_silos, y_silos

    # simply select test silos from saved silos
    def getTest(self, disease):
        dd = {'Atelectasis':5, 
            'Consolidation':3,
            'LungLesion':1,
            'LungOpacity':7,
            'PleuralEffusion':6,
            'PleuralOther':0,
            'Pneumonia':4,
            'Pneumothorax':2}

        selected_cls = dd[disease]
        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("test data and label reading finished!")
        else:
            print("test set not exist!")
        return x_silos, y_silos