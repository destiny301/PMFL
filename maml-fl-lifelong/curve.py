# =====Destiny======
# plot results(include MAML training in each round)

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay
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
    xtest, ytest = torch.from_numpy(xtest).to(device), torch.from_numpy(ytest).float()
    print("Test Data and Label shape:", xtest.shape, ytest.shape)

    times = 5 # running times for average to compute mean and std
    numOfAlgo = 4
    color = ['k', 'b', 'g', 'r'] # plot color
    label = ['w/o FL', 'w/ FL', 'MetaFL', 'PMFL'] # plot label
    auc = np.zeros([args.epoch_te, 1], dtype = float)
    pr = np.zeros([args.epoch_te, 1], dtype = float)

    # train and test, compute test auc
    
    model = Model(lib_sz, args.n_way).to(device)
    PATH = os.path.join(folder, 'PMFL/'+args.disease+'/model/01pmfl.pt')
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    criterion = nn.BCELoss()

    # compare 3 cases: w/o MAML, w/ MAML, part-freeze MAML
    training_loss = 0.0
    for epoch in range(args.epoch_te):
        db_t = DataLoader(db_test, args.k_te, shuffle=True, num_workers=1, pin_memory=True) # getitem() only return xtr(training) data
        for xtr, ytr in db_t:
            xtr, ytr = xtr.to(device), ytr.float().to(device)
            l = [text_length]*(xtr.shape[0])
            optimizer.zero_grad()
            # pylint: disable=not-callable
            
            logits = model(xtr, torch.tensor(l)).flatten()
            loss = criterion(logits, ytr)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        # compute test auc
        with torch.no_grad():
            l = [text_length]*(xtest.shape[0])
            logits_te = model(xtest, torch.tensor(l)).flatten()
            # pred_q = logits_te.argmax(dim=1)
            
            try:
                auc[epoch] = roc_auc_score(ytest, logits_te.cpu())
                pr[epoch] = average_precision_score(ytest, logits_te.cpu())
            except ValueError:
                pass
        print('epoch:', epoch+1, '\ttraining loss:', training_loss/len(db_t), '\ttest AUC:', auc[epoch], '\tPr:', pr[epoch])
        training_loss = 0.0
    
    fpr, tpr, th = roc_curve(ytest, logits_te.cpu())
    precision, recall, thresholds = precision_recall_curve(ytest, logits_te.cpu())
    optimal_idx = np.argmax(tpr - fpr)
    print(optimal_idx)
    print(fpr[optimal_idx], tpr[optimal_idx], th[optimal_idx])
    print(precision.shape)
    f = np.multiply(precision, recall)
    f= np.divide(2*f, (precision+recall))
    optimal_idx = np.argmax(f)
    print(optimal_idx, f[optimal_idx])
    print(precision[optimal_idx], recall[optimal_idx], thresholds[optimal_idx])


    plt.plot(fpr, tpr, 'b', label = 'ROC')
    plt.plot(precision, recall, 'r', label = 'Precision-Recall Curve')
    plt.legend()
    plt.title('ROC vs Precision-Recall Curve')
    # plt.show()
    
            
    imgPATH = os.path.join(folder, 'PMFL/roc_pr.png')
    plt.savefig(imgPATH)

    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch_te', type=int, help='epoch number for test task', default=2)

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
                            LungOpacity, PleuralEffusion, PleuralOther, Pneumonia, Pneumothorax)', default='PleuralOther')

    args = argparser.parse_args()

    main(args)