# Destiny
# fedAvg algorithm

import  torch
from    torch import nn
from    torch import optim
import  numpy as np
from    model import Model


class FL(nn.Module):
    def __init__(self, args, lib_sz, device, text_length):
        super(FL, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.epoch = args.epoch
        self.epoch_te = args.epoch_te
        self.device = device
        self.text_length = text_length
        self.lib_sz = lib_sz
        self.n_way = args.n_way
        self.task_num = args.n_silo

        self.net = Model(self.lib_sz, self.n_way)
        self.para = self.getCopy()
        self.subp = []
        for _ in range(self.task_num):
            self.subp.append(self.para)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.criterion = nn.BCELoss()

        self.weight = [1]*self.task_num
        self.n_batch = args.n_batch

    def loadParameters(self, para, model):
        p = []
        for index, param in enumerate(para):
            p.append(param.data)
        model.embedding.weight.data = p[0]
        model.lstm.weight_ih_l0.data = p[1]
        model.lstm.weight_hh_l0.data = p[2]
        model.lstm.bias_ih_l0.data = p[3]
        model.lstm.bias_hh_l0.data = p[4]
        model.lstm.weight_ih_l0_reverse.data = p[5]
        model.lstm.weight_hh_l0_reverse.data = p[6]
        model.lstm.bias_ih_l0_reverse.data = p[7]
        model.lstm.bias_hh_l0_reverse.data = p[8]
        model.fc.weight.data = p[9]
        model.fc.bias.data = p[10]

    def getCopy(self):
        p = nn.ParameterList()
        for index, param in enumerate(self.net.parameters()):
            p.append(nn.Parameter(param.clone()))
       
        return p

    def getParameters(self):
        p = nn.ParameterList()
        for index, param in enumerate(self.net.parameters()):
            p.append(param)
       
        return p
        
    def getState(self):
        return self.net.state_dict()

    def getBatch(self, xtr, xte):
        xtrain = []
        xtest = []
        bsztr = xtr.shape[0]//self.n_batch
        bszte = xte.shape[0]//self.n_batch
        for i in range(self.n_batch):
            xtrain.append(xtr[i*bsztr:(i+1)*bsztr, :])
            xtest.append(xte[i*bszte:(i+1)*bszte, :])
        xtrain = np.array(xtrain)
        xtest = np.array(xtest)
        return xtrain, xtest
    
    def forward(self, xtr, ytr):
        losses_te = [0 for _ in range(self.epoch+1)]  # losses[i] is the loss on step i
        netPara = self.getCopy()

        Vars = []
        for i in range(self.task_num):
            self.loadParameters(self.subp[i], self.net) # in fact, need to load server model, and here transfer the server model to clients
            
            # pylint: disable=no-member
            x, y = torch.from_numpy(np.array(xtr[i])).to(self.device), torch.from_numpy(np.array(ytr[i])).float().to(self.device)
            xtrain = x[:int((list(x.size())[0]/2)), :]
            xtest = x[int((list(x.size())[0]/2)):, :]
            ytrain = y[:int((list(x.size())[0]/2))]
            ytest = y[int((list(x.size())[0]/2)):]
            l = [self.text_length]*(xtrain.shape[0])
            # 1. run the i-th task and compute loss for k=0
            # pylint: disable=not-callable
            logits = self.net(xtrain, torch.tensor(l)).flatten()
            loss = self.criterion(logits, ytrain)
            para = self.getParameters()
            grad = torch.autograd.grad(loss, para)
            # todo: don't use fast_weights, directly transfer parameters
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, para)))

            with torch.no_grad():
                l = [self.text_length]*(xtest.shape[0])
                logits_q = self.net(xtest, torch.tensor(l)).flatten()
                loss_q = self.criterion(logits_q, ytest)
                losses_te[0] += loss_q*self.weight[i]

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                self.loadParameters(fast_weights, self.net)
                l = [self.text_length]*(xtest.shape[0])
                logits_q = self.net(xtest, torch.tensor(l)).flatten()
                loss_q = self.criterion(logits_q, ytest)
                losses_te[1] += loss_q*self.weight[i]

            for k in range(1, self.epoch):
                # 1. run the i-th task and compute loss for k=1~K-1                
                l = [self.text_length]*(xtrain.shape[0])
                logits = self.net(xtrain, torch.tensor(l)).flatten()
                loss = self.criterion(logits, ytrain)
                # 2. compute grad on theta_pi
                fast_weights = self.getParameters()
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                self.loadParameters(fast_weights, self.net)
                l = [self.text_length]*(xtest.shape[0])
                logits_q = self.net(xtest, torch.tensor(l)).flatten()
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.criterion(logits_q, ytest)
                losses_te[k + 1] += loss_q*self.weight[i]

            mVars = self.net.state_dict()
            var = []
            for param in mVars:
                var.append(self.net.state_dict()[param])
            Vars.append(var)

        loss_te = losses_te[-1] / np.sum(self.weight)
        
        # average clients model
        for i in range(len(Vars)):
            Vars[i] = np.array(Vars[i]) * self.weight[i] / sum(self.weight)
        Vars = np.sum(Vars, axis=0)
        
        for i, param in enumerate(mVars):
            mVars[param] = Vars[i]
        self.net.load_state_dict(mVars)

        # transfer model to clients
        for i in range(self.task_num):
            self.subp[i] = self.getCopy()

        return loss_te