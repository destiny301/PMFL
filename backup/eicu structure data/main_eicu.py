# =====Destiny======
# train, test

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score
from backup.metamodel import Model
# from metaFL import MAML
# from FederatedLearning import FL
from eicufl import FL
from eicumfl import MAML
from eicupmfl import PMFL
from data.dataloader import batchloader
from data.mimic_reader import mimicReader
from data.eicu_reader import EICUReader
import os

def main(args):
    # folder = '../../Dataset/mimic'
    folder = '../../datasets/eicu'
    PATH = os.path.join(folder, 'model/3'+args.algo+'.pt')
    # PATH = os.path.join(folder, args.disease+'/model/05'+args.algo+'.pt')

    # torch.manual_seed(111)
    # torch.cuda.manual_seed_all(111)
    # np.random.seed(111)
    print(args)
    # pylint: disable=no-member
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read all files in the folder
    print("=============Start Files Reading============")
    # dataset = FederatedReader(folder)
    dataset = EICUReader(folder)
    lib_sz = dataset.get_lib_sz() # get the number of features, used for the first layer of model
    text_length = 0
    # transform data into silos(train and test task), and get the star data of each silo to compute similarity
    # xte, yte = dataset.getOriginSplit()
    dataset.split2save()
    xte, yte, clste = dataset.getTest(args.disease)
    # xte, yte = dataset.testdata()
    print(yte.shape)
    print(np.count_nonzero(yte == 0))
    print(np.count_nonzero(yte == 1))
    # print(np.count_nonzero(yte == 2))
    # print(np.count_nonzero(yte == 3))
    # print(np.count_nonzero(yte == 8))
    x_silos, y_silos, clstr = dataset.getTrain(args.n_silo, args.disease)
    # xte, yte = dataset.getSilos(1, 'test')
    # x_silos, y_silos = dataset.getSilos(args.n_silo, 'train')
    
    # generate dataset for model training
    print("===========Generate dataset=========")
    xtr, ytr = batchloader(x_silos, y_silos, ratio = args.ratio).createDataset(args.n_silo, args.n_batch)
    db_test = batchloader(xte, yte, ratio = args.ratio, mode = 'test') # for test task, choose the latter half for test data, and ratio for training data
    
    # create model
    print("===========Create Model===========")
    testModel = Model(lib_sz, clste).to(device)
    if args.algo == 'fl':
        trainModel = FL(args, lib_sz, device, text_length).to(device)
    elif args.algo == 'mfl':
        trainModel = MAML(args, lib_sz, device, text_length).to(device)
    else:
        trainModel = PMFL(args, lib_sz, device, text_length, clstr, clste).to(device)

    optimizer = optim.Adam(testModel.parameters(), lr=args.meta_lr)
    criterion = nn.BCELoss()

    tmp = filter(lambda x: x.requires_grad, trainModel.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(trainModel)
    print('Total trainable tensors:', num)

    # start training train tasks
    print("===========Train training tasks==========")
    for step in range(args.n_step):
        # print('--------------Round {}---------------'.format(step+1))
        losses = 0.0
        for bn in range(args.n_batch):
            x = xtr[bn]
            y = ytr[bn]
            loss = trainModel(x, y)
            losses += loss.item()
        print('round:', step+1, '\ttraining loss:', losses/args.n_batch)
    torch.save(trainModel.getState(), PATH)

    print("===========Train New Task===========")
    # trainModel.loadModel(trainModel.getCopy(), testModel)'model01.pt'
    # testModel.load_state_dict(torch.load(PATH))
    testModel.load_state_dict(trainModel.getState())
    training_loss = 0.0
    xtest, ytest = db_test.get_testdata()
    # print(ytest.shape)
    # print(np.count_nonzero(ytest == 0))
    # print(np.count_nonzero(ytest == 1))
    # print(np.count_nonzero(ytest == 2))
    # print(np.count_nonzero(ytest == 3))
    # print(np.count_nonzero(ytest == 8))
    # print(ytest[:10], ytest.shape())
    xtest, ytest = torch.from_numpy(xtest).to(device), torch.from_numpy(ytest).float()
    print("Test Data and Label shape:", xtest.shape, ytest.shape)
    acc = 0.0
    auc = 0.0
    correct_pred = [0, 0]
    total_pred = [0, 0]
    for epoch in range(args.epoch_te):
        db_t = DataLoader(db_test, args.k_te, shuffle=True, num_workers=1, pin_memory=False)
        for xtr, ytr in db_t:
            xtr, ytr = xtr.to(device), ytr.float().to(device)
            l = [text_length]*(xtr.shape[0])
            optimizer.zero_grad()
            # pylint: disable=not-callable
            logits = testModel(xtr, torch.tensor(l)).flatten()
            # print(logits)
            # print(logits.squeeze())
            # print(ytr)
            loss = criterion(logits, ytr)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        with torch.no_grad():
            l = [text_length]*(xtest.shape[0])
            # pylint: disable=not-callable
            logits_te = testModel(xtest, torch.tensor(l)).cpu().flatten()
            # pred_q = logits_te.argmax(dim=1)
            # print(pred_q)
            # print(pred_q.size()[0])
            # print(torch.eq(pred_q, ytest).sum().item())
            # print(ytest)
            # acc = torch.eq(pred_q, ytest).sum().item()/pred_q.size()[0]
            # for label, prediction in zip(ytest, pred_q):
            #     if label == prediction:
            #         correct_pred[label] += 1
            #     total_pred[label] += 1
            try:
                auc = roc_auc_score(ytest, logits_te.cpu())
            except ValueError:
                pass
        print('epoch:', epoch+1, '\ttraining loss:', training_loss/len(db_t), '\tauc:', auc)
        # print('epoch:', epoch+1, '\ttraining loss:', training_loss/len(db_t), '\tTest acc:', acc, '\tnum:', np.count_nonzero(pred_q==1), np.count_nonzero(ytest==1),
        #         '\tacc:', correct_pred[0]/total_pred[0], correct_pred[1]/total_pred[1])
        acc = 0.0
        correct_pred = [0, 0]
        total_pred = [0, 0]
        # print('epoch:', epoch+1, '\ttraining loss:', training_loss/len(db_t))
        training_loss = 0.0
    # print("============Test New Task=============")
    # xtest, ytest = db_test.get_testdata()
    # xtest, ytest = torch.from_numpy(xtest).to(device), torch.from_numpy(ytest).float()
    # print("Test Data and Label shape:", xtest.shape, ytest.shape)
    # auc = 0.0
    # with torch.no_grad():
    #     l = [text_length]*(xtest.shape[0])
    #     # pylint: disable=not-callable
    #     logits_te = testModel(xtest, torch.tensor(l)).squeeze()
    #     # pred_q = logits_te.argmax(dim=1)
    #     try:
    #         auc = roc_auc_score(ytest, logits_te.cpu())
    #     except ValueError:
    #         pass
    # print('Test ROC AUC Score:', auc)
    
    

    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20)
    argparser.add_argument('--epoch_te', type=int, help='epoch number for test task', default=10)

    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    # argparser.add_argument('--k_tr', type=int, help='k shot for train set', default=10)
    argparser.add_argument('--k_te', type=int, help='k shot for test set', default=128)
    argparser.add_argument('--meta_lr', type=float, help='meta-level learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='learning rate', default=0.01)

    # federated learning
    argparser.add_argument('--n_silo', type=int, help='num of data sources', default=3)
    argparser.add_argument('--n_batch', type=int, help='num of batches', default=50)
    argparser.add_argument('--n_step', type=int, help='num of all sources ave update', default=20)
    argparser.add_argument('--ratio', type=float, help='ratio of training data in test silo', default=0.8)
    # maml or part-freeze maml
    argparser.add_argument('--algo', type=str, help='choose Federated Learning(fl), maml-Federated Learning(mfl) \
                            or Partial Meta-Federated Learning(pmfl)', default='fl')
    argparser.add_argument('--disease', type=str, help='choose target task(Atelectasis, Consolidation, LungLesion,\
                            LungOpacity, PleuralEffusion, PleuralOther, Pneumonia, Pneumothorax)', default='Consolidation')
    args = argparser.parse_args()

    main(args)