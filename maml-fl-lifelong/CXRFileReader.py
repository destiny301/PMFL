# =====Destiny======
# read data and split

import os
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import re, io
import torch
from torch.nn.utils.rnn import pad_sequence

class FederatedReader():

    def __init__(self, root):

        datafolder = os.path.join(root, 'files')
        labelpath = os.path.join(root, 'mimic-cxr-2.0.0-chexpert.csv')
        datasetpath = os.path.join('../../Dataset/mimic', 'mimicDataset.npz')
        self.testsetpath = os.path.join('.../../Dataset/mimic/Pneumothorax', 'test.npz')
        self.trainsetpath = os.path.join('../../Dataset/mimic/Pneumothorax', 'train.npz')
        self.datasilospath = os.path.join('../../Dataset/mimic', 'datasilo.npz')
        self.data = []
        self.label = []
        self.tensor = []
        self.text2index = {}
        self.index2text = {}
        self.vocab_sz = 0

        if os.path.exists(datasetpath):
            dataset = np.load(datasetpath, allow_pickle=True)
            self.tensor = dataset['tensor']
            self.label = dataset['label']
            self.text2index = dataset['text2index'].item()
            self.index2text = dataset['index2text'].item()
            self.vocab_sz = dataset['vocab_sz']
            print("data and label reading finished!")
            print("vocab_sz:", self.vocab_sz)
        else:
            print("Dataset not existed, need to read!")
            df = pd.read_csv(labelpath, usecols=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 
                        'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax'])
            df.fillna(0, inplace=True)
            array = df.to_numpy()
            self.label = np.where(array==-1, 0, array).astype(np.int64)
            print("labels reading finished!")
            for dir1 in sorted(os.listdir(datafolder)):
                dir1path = os.path.join(datafolder, dir1)
                for dir2 in sorted(os.listdir(dir1path)):
                    dir2path = os.path.join(dir1path, dir2)
                    for files in sorted(os.listdir(dir2path)):
                        filepath = os.path.join(dir2path, files)
                        lines = io.open(filepath, encoding='UTF-8').read().strip().split('\n')
                        self.data.append(self.preprocess(' '.join(lines)))
            print("data reading finished!")
            print(len(self.data))
            print(self.data[0])
            self.tokenizer(self.data)
            # save dataset as a numpy array
            np.savez(datasetpath, tensor=self.tensor, label=self.label, 
                        text2index = self.text2index, index2text = self.index2text, vocab_sz = self.vocab_sz)
            print("data saving finished!")
        self.tag = [-1]*self.tensor.shape[0]
        # self.cls_num = 16
        self.tensor, self.label = shuffle(self.tensor, self.label)
        print("Total data and Label shape:", self.tensor.shape, self.label.shape)

    def preprocess(self, s):

        s = re.sub(r"([?!.,¿])", r" \1", s)
        s = re.sub(r'[" "]+', " ", s)

        s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
        
        s = s.strip()
        return s

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
            # pylint: disable=no-member
            id = torch.zeros(len(text), dtype = torch.int)
            for j in range(len(text)):
                id[j] = self.text2index[text[j]]
            self.tensor.append(id)
        self.tensor = pad_sequence(self.tensor, batch_first = True, padding_value = 0)
        self.tensor = np.asarray(self.tensor)

    def get_lib_sz(self):
        return self.vocab_sz
    def get_text_length(self):
        return self.tensor.shape[1]

    def getSilos(self, n_silo=10, mode='train'):
        x_silos = []
        y_silos = []

        if mode=='train':
            classes = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
            # selected_cls = np.random.choice(classes, n_silo, False)
            selected_cls = [6, 9, 0, 2, 10, 11, 7, 8]
            for cur_class in selected_cls:
                x_silo = []
                y_silo = []
                for index,labels in enumerate(self.label):
                    if labels[cur_class] == 1 and self.tag[index] == -1:
                        x_silo.append(self.tensor[index])
                        y_silo.append(1)
                        self.tag[index] = cur_class

                class_sz = len(y_silo)*2
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
            np.savez(self.trainsetpath, x=x_silos, y=y_silos)
            print(x_silos.shape, x_silos.dtype)
        else:
            cur_class = 1
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 1 and self.tag[index] == -1:
                    x_silos.append(self.tensor[index])
                    y_silos.append(1)
                    self.tag[index] = 1

            class_sz = len(y_silos)*2
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 0 and self.tag[index] == -1:
                    x_silos.append(self.tensor[index])
                    y_silos.append(0)
                    self.tag[index] = 1
                if len(y_silos) == class_sz:
                    break
            x_silos, y_silos = shuffle(x_silos, y_silos)
            x_silos = np.array(x_silos).astype(np.int64)
            y_silos = np.array(y_silos).astype(np.int64)
            np.savez(self.testsetpath, x=x_silos, y=y_silos)
            print("Test class shape:", x_silos.shape, y_silos.shape)
        return x_silos, y_silos

    def split2save(self):
        x_silos = []
        y_silos = []
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # desease class 16 has no patients
        # selected_cls = np.random.choice(classes, n_silo, False)
        selected_cls = [9, 6, 11, 2, 10, 0, 8, 7]
        for cur_class in selected_cls:
            x_silo = []
            y_silo = []
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 1 and self.tag[index] == -1:
                    x_silo.append(self.tensor[index])
                    y_silo.append(1)
                    self.tag[index] = cur_class
            class_sz = len(y_silo)*2
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
        print(x_silos.shape, x_silos.dtype)

    def getTrain(self, n_silo, disease):
        # classes = [0, 1, 2, 3, 4, 5, 6, 7]
        # selected_cls = np.random.choice(classes, n_silo, False)
        dd = {'Atelectasis':[1, 2, 3, 4, 7], 
            'Consolidation':[6, 0, 7, 2, 5],
            'LungLesion':[6, 0, 7, 2, 5],
            'LungOpacity':[6, 0, 1, 3, 5],
            'PleuralEffusion':[1, 2, 3, 4, 7],
            'PleuralOther':[1, 2, 3, 4, 7],
            'Pneumonia':[6, 0, 7, 2, 5],
            'Pneumothorax':[6, 0, 1, 3, 5]}

        selected_cls = dd[disease]

        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("train data and label reading finished!")
        else:
            print("train set not exist!")
        return x_silos, y_silos

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