# =====Destiny======
# train, test

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score
from models.model import Model
from models.metaFL import MAML
from models.FederatedLearning import FL
from models.pmfl import PMFL
from data.dataloader import batchloader
from data.mimic_reader import mimicReader
from data.eicu_reader import EICUReader
import os

def main(args):
    folder = os.path.join('../datasets', args.data)
    PATH = os.path.join(folder, args.disease+'/model/01'+args.algo+'.pt') # saving model path

    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)
    np.random.seed(111)
    print(args)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # read all files in the folder
    print("=============Start Files Reading============")
    if args.data == 'mimic':
        dataset = mimicReader(folder)
    else:
        dataset = EICUReader(folder)
    lib_sz = dataset.get_lib_sz() # get the number of features, used for the first layer of model
    text_length = dataset.get_text_length()
    # transform data into silos(train and test task)
    if not os.path.exists(os.path.join(folder, 'data/datasilo.npz')):
        dataset.split2save()
    xte, yte = dataset.getTest(args.disease)
    x_silos, y_silos = dataset.getTrain(args.n_silo, args.disease)
    
    # generate dataset for model training
    print("===========Generate dataset=========")
    xtr, ytr = batchloader(x_silos, y_silos, ratio = args.ratio).createDataset(args.n_silo, args.n_batch)
    db_test = batchloader(xte, yte, ratio = args.ratio, mode = 'test') 
    
    # create model
    print("===========Create Model===========")
    testModel = Model(lib_sz, args.n_way).to(device)
    if args.algo == 'fl':
        trainModel = FL(args, lib_sz, device, text_length).to(device)
    elif args.algo == 'mfl':
        trainModel = MAML(args, lib_sz, device, text_length).to(device)
    else:
        trainModel = PMFL(args, lib_sz, device, text_length).to(device)

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
    # trainModel.loadModel(trainModel.getCopy(), testModel)
    # testModel.load_state_dict(torch.load(PATH))
    testModel.load_state_dict(trainModel.getState()) # load the pretrained model into the server model
    training_loss = 0.0
    for epoch in range(args.epoch_te):
        db_t = DataLoader(db_test, args.k_te, shuffle=True, num_workers=1, pin_memory=True)
        for xtr, ytr in db_t:
            xtr, ytr = xtr.to(device), ytr.float().to(device)
            l = [text_length]*(xtr.shape[0])
            optimizer.zero_grad()
            # pylint: disable=not-callable
            logits = testModel(xtr, torch.tensor(l)).flatten()
            loss = criterion(logits, ytr)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        print('epoch:', epoch+1, '\ttraining loss:', training_loss/len(db_t))
        training_loss = 0.0

    print("============Test New Task=============")
    xtest, ytest = db_test.get_testdata()
    xtest, ytest = torch.from_numpy(xtest).to(device), torch.from_numpy(ytest).float()
    print("Test Data and Label shape:", xtest.shape, ytest.shape)
    auc = 0.0
    with torch.no_grad():
        l = [text_length]*(xtest.shape[0])
        logits_te = testModel(xtest, torch.tensor(l)).flatten()
        # pred_q = logits_te.argmax(dim=1)
        try:
            auc = roc_auc_score(ytest, logits_te.cpu())
        except ValueError:
            pass
    print('Test ROC AUC Score:', auc)

    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch_te', type=int, help='epoch number for test task', default=10)

    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    # argparser.add_argument('--k_tr', type=int, help='k shot for train set', default=10)
    argparser.add_argument('--k_te', type=int, help='k shot for test set', default=20)
    argparser.add_argument('--meta_lr', type=float, help='meta-level learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='learning rate', default=0.01)

    # federated learning
    argparser.add_argument('--n_silo', type=int, help='num of data sources', default=5)
    argparser.add_argument('--n_batch', type=int, help='num of batches', default=100)
    argparser.add_argument('--n_step', type=int, help='num of all sources ave update', default=20)
    argparser.add_argument('--ratio', type=float, help='ratio of training data in test silo', default=0.9)
    # maml or part-freeze maml
    argparser.add_argument('--algo', type=str, help='choose Federated Learning(fl), maml-Federated Learning(mfl) \
                            or Partial Meta-Federated Learning(pmfl)', default='fl')
    argparser.add_argument('--disease', type=str, help='choose target task(Atelectasis, Consolidation, LungLesion,\
                            LungOpacity, PleuralEffusion, PleuralOther, Pneumonia, Pneumothorax)', default='Consolidation')
    
    argparser.add_argument('--data', type=str, help='which dataset(mimic or eicu)', default='mimic')
    args = argparser.parse_args()

    main(args)