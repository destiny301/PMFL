# =====Destiny======
# read data and split

import os
import numpy as np
import random
from sklearn.utils import shuffle
import re, io
import torch

from torch.nn.utils.rnn import pad_sequence

class I2B2Reader():

    def __init__(self, root):

        self.path = root
        folder = '../../datasets/challenge2008/training'
        # folder = '../../datasets/challenge2008/test'
        datafolder = os.path.join(root, 'notes')
        labelfolder = os.path.join(root, 'labels')
        datasetpath = os.path.join(root, 'mimicDataset.npz')
        # self.testsetpath = os.path.join('.../../Dataset/mimic/Pneumothorax', 'test.npz')
        # self.trainsetpath = os.path.join('../../Dataset/mimic/Pneumothorax', 'train.npz')
        self.datasilospath = os.path.join(root, 'datasilo.npz')

        self.data = []
        self.label = []
        self.vocab_sz = 0
        
        deseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'GERD', 'Gallstones',
                    'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia',
                    'OA', 'OSA', 'Obesity', 'PVD', 'Venous Insufficiency'] # 5, 7, 8, 9, 10, 13(obesity)

        if os.path.exists(datasetpath):
            dataset = np.load(datasetpath, allow_pickle=True)
            self.tensor = dataset['tensor']
            self.label = dataset['label']

            self.vocab_sz = dataset['vocab_sz']
            print("data and label reading finished!")
            print("vocab_sz:", self.vocab_sz)
        else:
            print("Dataset not existed, need to read!")
            for desease in deseases:
                label_d = []
                labelfolder = os.path.join(root, 'labels')
                for file in os.listdir(labelfolder):
                    with open(os.path.join(labelfolder, file),'r') as file_read:
                        y = 0
                        for line in file_read.readlines():
                            line = line.split()
                            # print(line)
                            if line[0] == 'intuitive':
                                if line[2] == desease:
                                    y = 1 if line[4] == 'Y' else 0
                        label_d.append(y)
                labelfolder = os.path.join(folder, 'labels')
                for file in os.listdir(labelfolder):
                    with open(os.path.join(labelfolder, file),'r') as file_read:
                        y = 0
                        for line in file_read.readlines():
                            line = line.split()
                            # print(line)
                            if line[0] == 'intuitive':
                                if line[2] == desease:
                                    y = 1 if line[4] == 'Y' else 0
                        label_d.append(y)
                self.label.append(label_d)

            self.label = np.transpose(np.array(self.label))

            for file in os.listdir(datafolder):
                filepath = os.path.join(datafolder, file)
                lines = io.open(filepath, encoding='UTF-8').read().strip().split('\n')
                self.data.append(self.preprocess(' '.join(lines)))

            datafolder = os.path.join(folder, 'notes')
            for file in os.listdir(datafolder):
                filepath = os.path.join(datafolder, file)
                lines = io.open(filepath, encoding='UTF-8').read().strip().split('\n')
                self.data.append(self.preprocess(' '.join(lines)))
            # self.data = self.data*16
            self.cls_num = 16
            self.tokenizer(self.data)
            np.savez(datasetpath, tensor=self.tensor, label=self.label, vocab_sz = self.vocab_sz)
            print("data saving finished!")
        self.tag = [-1]*self.tensor.shape[0]
        self.xo = self.tensor
        self.yo = self.label

        self.tensor, self.label = shuffle(self.tensor, self.label)
        print(self.tensor.shape, self.label.shape)

    def getOriginSplit(self):
        return self.xo, self.yo[:, 13]
    
        
    def preprocess(self, s):

        s = re.sub(r"([?!,¿])", r" \1", s)
        s = re.sub(r'[" "]+', " ", s)

        s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
        
        s = s.strip()
        return s

    def tokenizer(self, data):
        self.tensor = []
        lib = []
        for i in range(len(data)):
            text = data[i].split()
            # print(len(text))
            for j in range(len(text)):
                if text[j] not in lib:
                    lib.append(text[j])

        libdic = {m:n for n, m in enumerate(lib)}
        self.vocab_sz = len(lib)
        for i in range(len(data)):
            text = data[i].split()
            # print(len(text))
            # pylint: disable=no-member
            id = torch.zeros(len(text), dtype = torch.int)
            for j in range(len(text)):
                id[j] = libdic[text[j]]
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
            # to do: choose randomly class in loop, or outside loop?
            # classes = np.arange(1, 16)
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15] 
            # selected_cls = np.random.choice(classes, n_silo, False)
            selected_cls = [2, 12, 14]
            for cur_class in selected_cls:
                x_silo = []
                y_silo = []
                for index,labels in enumerate(self.label):
                    if labels[cur_class] == 1 and self.tag[index] == -1:
                        x_silo.append(self.tensor[index])
                        y_silo.append(1)
                        self.tag[index] = cur_class
                    if len(y_silo) >= 100:
                        break
                # to do: 1:1, or 1:2?
                class_sz = len(y_silo)*2
                # print(len(y_silo))

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
            print("train data shape:", x_silos.shape, y_silos.shape)
        else:
            cur_class = 13
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 1 and self.tag[index] == -1:
                    x_silos.append(self.tensor[index])
                    y_silos.append(1)
                    self.tag[index] = 13
                if len(y_silos) >= 200:
                    break

            class_sz = len(y_silos)*2
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 0 and self.tag[index] == -1:
                    x_silos.append(self.tensor[index])
                    y_silos.append(0)
                    self.tag[index] = 13
                if len(y_silos) == class_sz:
                    break
            x_silos, y_silos = shuffle(x_silos, y_silos)
            x_silos = np.array(x_silos).astype(np.int64)
            y_silos = np.array(y_silos).astype(np.int64)
            print("test class shape:", x_silos.shape, y_silos.shape)

        return x_silos, y_silos

    def gettestdata(self):
        return self.tensor, self.label[:, 4]
    def split2save(self):
        x_silos = []
        y_silos = []
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]  # desease class 16 has no patients
        # selected_cls = np.random.choice(classes, n_silo, False)
        selected_cls = [10, 13, 7, 5, 8, 9]
        limit = 80
        r = 2
        for cur_class in selected_cls:
            x_silo = []
            y_silo = []
            for index,labels in enumerate(self.label):
                if labels[cur_class] == 1 and self.tag[index] == -1:
                    x_silo.append(self.tensor[index])
                    y_silo.append(1)
                    self.tag[index] = cur_class
                if cur_class == 13:
                    limit = 400
                    r = 2
                else:
                    limit = 200
                    r = 2
                if len(y_silo) >= limit:
                    break
            class_sz = len(y_silo)*r
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
        # dd = {'Atelectasis':[1, 2, 3, 4, 7], 
        #     'Consolidation':[6, 0, 7, 2, 5],
        #     'LungLesion':[6, 0, 7, 2, 5],
        #     'LungOpacity':[6, 0, 1, 3, 5],
        #     'PleuralEffusion':[1, 2, 3, 4, 7],
        #     'PleuralOther':[1, 2, 3, 4, 7],
        #     'Pneumonia':[6, 0, 7, 2, 5],
        #     'Pneumothorax':[6, 0, 1, 3, 5]}

        selected_cls = [0, 2, 3, 4, 5]

        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("train data and label reading finished!")
        else:
            print("train set not exist!")
        return x_silos, y_silos

    def getTest(self, disease):
        # dd = {'Atelectasis':5, 
        #     'Consolidation':3,
        #     'LungLesion':1,
        #     'LungOpacity':7,
        #     'PleuralEffusion':6,
        #     'PleuralOther':0,
        #     'Pneumonia':4,
        #     'Pneumothorax':2}

        selected_cls = 1
        if os.path.exists(self.datasilospath):
            dataset = np.load(self.datasilospath, allow_pickle=True)
            x_silos = dataset['x'][selected_cls]
            y_silos = dataset['y'][selected_cls]
            print("test data and label reading finished!")
        else:
            print("test set not exist!")

        return x_silos, y_silos