# =====Destiny======
# plot results(include MAML training in each round)

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from model import Model
from datagenerator import I2B2Dataset
from CXRFileReader import FederatedReader
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main(args):

    folder = '../../Dataset/mimic' # data and model path
    print(args)
    # pylint: disable=no-member
    # pylint: disable=not-callable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    numOfAlgo = 4
    aucPATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/05'+args.disease+'_rocauc.npy') # 03--include maml training in each round, 04-hald data
    prPATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/05'+args.disease+'_prauc.npy') 
    f1PATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/05'+args.disease+'_f1.npy') # 03--include maml training in each round, 04-hald data
    pPATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/05'+args.disease+'_precision.npy') 
    rPATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/05'+args.disease+'_recall.npy') 
    auc = np.load(aucPATH)
    pr = np.load(prPATH)
    f1 = np.load(aucPATH)
    p = np.load(prPATH)
    r = np.load(aucPATH)

    color = ['k', 'b', 'g', 'r'] # plot color
    label = ['w/o FL', 'w/ FL', 'MetaFL', 'PMFL'] # plot label
    # algo = ['w/o FL', 'w/ FL', 'MetaFL', 'PMFL'] 
    x = np.arange(args.epoch_te)+1
    fig, ax = plt.subplots()
    with sns.axes_style("darkgrid"):
        for i in range(numOfAlgo):
            mean = np.mean(auc[i], 0)
            std = np.std(auc[i], 0)
            ax.plot(x, mean, color[i], label = label[i])
            ax.fill_between(x, mean-std, mean+std, facecolor = color[i], alpha = 0.2)
            print("i={:d}, {:.4f}, {:.4f}".format(i+1, mean[9], std[9]))
        ax.legend()
    
    plt.xlabel('epoch')
    plt.ylabel('Test AUC')
    plt.title(args.disease) # 5 silos
    imgPATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/01'+args.disease+'_rocauc.png')
    # plt.savefig(imgPATH)
    print('-----------')
    fig, ax = plt.subplots()
    with sns.axes_style("darkgrid"):
        for i in range(numOfAlgo):
            mean = np.mean(pr[i], 0)
            std = np.std(pr[i], 0)
            ax.plot(x, mean, color[i], label = label[i])
            ax.fill_between(x, mean-std, mean+std, facecolor = color[i], alpha = 0.2)
            print("i={:d}, {:.4f}, {:.4f}".format(i+1, mean[9], std[9]))
        ax.legend()
    plt.xlabel('epoch')
    plt.ylabel('Test Precision')
    plt.title(args.disease) # 5 silos
    imgPATH = os.path.join(folder, 'PMFL/'+args.disease+'/result/01'+args.disease+'_prauc.png')
    # plt.savefig(imgPATH)
    # plt.show()
    print('-----------')
    for i in range(numOfAlgo):
        mean = np.mean(f1[i], 0)
        std = np.std(f1[i], 0)
        print("i={:d}, {:.4f}, {:.4f}".format(i+1, mean[9], std[9]))
    print('-----------')
    for i in range(numOfAlgo):
        mean = np.mean(p[i], 0)
        std = np.std(p[i], 0)
        print("i={:d}, {:.4f}, {:.4f}".format(i+1, mean[9], std[9]))
    print('-----------')
    for i in range(numOfAlgo):
        mean = np.mean(r[i], 0)
        std = np.std(r[i], 0)
        print("i={:d}, {:.4f}, {:.4f}".format(i+1, mean[9], std[9]))
    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch_te', type=int, help='epoch number for test task', default=10)

    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    # argparser.add_argument('--k_tr', type=int, help='k shot for train set', default=10)
    argparser.add_argument('--k_te', type=int, help='k shot for test set', default=64)
    argparser.add_argument('--meta_lr', type=float, help='meta-level learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='learning rate', default=0.01)

    # federated learning
    argparser.add_argument('--n_silo', type=int, help='num of data sources', default=3)
    argparser.add_argument('--n_batch', type=int, help='num of batches', default=50)
    argparser.add_argument('--n_step', type=int, help='num of all sources ave update', default=20)
    argparser.add_argument('--ratio', type=float, help='ratio of training data in test silo', default=0.9)
    # maml or part-freeze maml
    argparser.add_argument('--algo', type=str, help='choose Federated Learning(fl), maml-Federated Learning(mfl) \
                            or Partial Meta-Federated Learning(pmfl)', default='fl')
    argparser.add_argument('--disease', type=str, help='choose target task(Atelectasis, Consolidation, LungLesion,\
                            LungOpacity, PleuralEffusion, PleuralOther, Pneumonia, Pneumothorax)', default='LungOpacity')

    args = argparser.parse_args()

    main(args)