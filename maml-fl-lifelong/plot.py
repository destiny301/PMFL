# =====Destiny======
# plot results(include MAML training in each round)

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score
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
    
    # read all files in the folder
    dataset = FederatedReader(folder)
    lib_sz = dataset.get_lib_sz() # get the number of features, used for the first layer of model
    text_length = dataset.get_text_length()
    # transform data into silos(train and test task)
    xte, yte = dataset.getTest(args.disease)
  
    # generate dataset for model training
    db_test = I2B2Dataset(xte, yte, ratio = args.ratio, mode = 'test') # for test task, choose the latter half for test data, and ratio for training data
    xtest, ytest = db_test.get_testdata()
    xtest, ytest = torch.from_numpy(xtest).to(device), torch.from_numpy(ytest).to(device)
    print("Test Data and Label shape:", xtest.shape, ytest.shape)

    times = 5 # running times for average to compute mean and std
    color = ['k', 'b', 'g'] # plot color
    label = ['w/o MAML', 'w/ MAML', 'part-freeze MAML'] # plot label
    auc = np.zeros([3, times, args.epoch_te], dtype = float)

    # train and test, compute test auc
    for t in range(times):
        print('==========Round', t+1, '============')
        # each round, initize a new model
        model = Model(lib_sz, args.n_way).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
        criterion = nn.CrossEntropyLoss()

        # compare 3 cases: w/o MAML, w/ MAML, part-freeze MAML
        for i in range(3):
            print("-----------", label[i], "------------")
            training_loss = 0.0
            for epoch in range(args.epoch_te):
                db_t = DataLoader(db_test, args.k_te, shuffle=True, num_workers=1, pin_memory=True) # getitem() only return xtr(training) data
                for xtr, ytr in db_t:
                    xtr, ytr = xtr.to(device), ytr.to(device)
                    l = [text_length]*(xtr.shape[0])
                    optimizer.zero_grad()
                    # pylint: disable=not-callable
                    logits = model(xtr, torch.tensor(l))
                    loss = criterion(logits, ytr)
                    loss.backward()
                    optimizer.step()

                    training_loss += loss.item()
                # compute test auc
                with torch.no_grad():
                    l = [text_length]*(xtest.shape[0])
                    logits_te = model(xtest, torch.tensor(l))
                    pred_q = logits_te.argmax(dim=1)
                    try:
                        auc[i, t, epoch] = roc_auc_score(pred_q.cpu(), ytest.cpu())
                    except ValueError:
                        pass
                print('epoch:', epoch+1, '\ttraining loss:', training_loss/len(db_t), '\ttest ROCAUC:', auc[i, t, epoch])
                training_loss = 0.0

            if i == 0:
                PATH = os.path.join(folder, args.disease+'/model/0'+str(t+1)+'maml.pt')
                model.load_state_dict(torch.load(PATH)) # load MAML model
            elif i == 1:
                PATH = os.path.join(folder, args.disease+'/model/0'+str(t+1)+'pf.pt')
                model.load_state_dict(torch.load(PATH)) # load part-freeze model
                

    x = np.arange(args.epoch_te)+1
    fig, ax = plt.subplots()
    aucPATH = os.path.join(folder, args.disease+'/result/04'+args.disease+'.npy') # 03--include maml training in each round, 04-hald data
    np.save(aucPATH, auc)
    with sns.axes_style("darkgrid"):
        for i in range(3):
            mean = np.mean(auc[i], 0)
            std = np.std(auc[i], 0)
            ax.plot(x, mean, color[i], label = label[i])
            ax.fill_between(x, mean-std, mean+std, facecolor = color[i], alpha = 0.2)
        ax.legend()
    plt.xlabel('epoch')
    plt.ylabel('Test data ROCAUC')
    plt.title(args.disease+'(5silos, half data)')
    imgPATH = os.path.join(folder, args.disease+'/result/04'+args.disease+'.png')
    plt.savefig(imgPATH)
    # plt.show()
    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch_te', type=int, help='epoch number for test task', default=15)

    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    # argparser.add_argument('--k_tr', type=int, help='k shot for train set', default=10)
    argparser.add_argument('--k_te', type=int, help='k shot for test set', default=64)
    argparser.add_argument('--meta_lr', type=float, help='meta-level learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='learning rate', default=0.01)

    # federated learning
    argparser.add_argument('--n_silo', type=int, help='num of data sources', default=3)
    argparser.add_argument('--n_batch', type=int, help='num of batches', default=50)
    argparser.add_argument('--n_step', type=int, help='num of all sources ave update', default=20)
    argparser.add_argument('--ratio', type=float, help='ratio of training data in test silo', default=0.5)
    # maml or part-freeze maml
    argparser.add_argument('--algo', type=str, help='choose maml or part-freeze maml', default='pf')
    argparser.add_argument('--disease', type=str, help='choose target task(Atelectasis, Consolidation, LungLesion,\
                            LungOpacity, PleuralEffusion, PleuralOther, Pneumonia, Pneumothorax)', default='Atelectasis')

    args = argparser.parse_args()

    main(args)