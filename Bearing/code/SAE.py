# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import sys
import json
import joblib

params = {}
params['train'] = '/usr/train.csv'
params['test'] = '/usr/test.csv'
params['saved'] = '模型保存路径'
params['USE_SPARSE'] = 1
params['Beta'] = 0.2
params['rho'] = 0.05
argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

    train_csv = pd.read_csv(params['train'])

    test_csv = pd.read_csv(params['test'])

    X1 = train_csv.drop(['label'], axis=1)

    X2 = test_csv.drop(['label'], axis=1)

    Y1 = train_csv.loc[:, ['label']]

    Y2 = test_csv.loc[:, ['label']]


    X_scaler = StandardScaler()
    X1_scaler = X_scaler.fit_transform(X1)
    X2_scaler = X_scaler.fit_transform(X2)

    BATCH_SIZE = 512

    torch_x_train1 = torch.from_numpy(X1_scaler)
    torch_y_train1 = torch.from_numpy(Y1.values)
    train1_dataset = Data.TensorDataset(torch_x_train1, torch_y_train1)
    train1_loader = DataLoader(train1_dataset, batch_size=BATCH_SIZE, shuffle=False)

    torch_x_train2 = torch.from_numpy(X2_scaler)
    torch_y_train2 = torch.from_numpy(Y2.values)
    train2_dataset = Data.TensorDataset(torch_x_train2, torch_y_train2)
    train2_loader = DataLoader(train2_dataset, batch_size=BATCH_SIZE, shuffle=False)


    class SAE(nn.Module):
        def __init__(self):
            super(SAE, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(166, 166),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(166, 166),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Linear(166, 166),
                nn.ReLU(),
                nn.Linear(166, 166),
                nn.ReLU(),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded


    def weight_init(m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(6.0 / (fan_in + fan_out))
            # m.weight.data.normal_(0.0, variance)
            m.weight.data.uniform_(-variance, variance)


    def kl_divergence(p, q)
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)

        s1 = torch.sum(p * torch.log(p / q))
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        return s1 + s2


    LR = 0.005
    SAEncoder = SAE()
    SAEncoder.apply(weight_init)
    if torch.cuda.is_available():
        SAEncoder = SAEncoder.cuda()
    optimizer = torch.optim.Adam(SAEncoder.parameters(), lr=LR, weight_decay=1e-8)
    loss_mse = nn.MSELoss()
    kl_loss = nn.KLDivLoss()

    y_loss = []
    for epoch in range(350):
        i = 0
        for data1, data2 in zip(train1_loader, train2_loader):
            batch_x1, batch_y1 = data1
            batch_x2, batch_y2 = data2

            if torch.cuda.is_available():
                batch_x1 = batch_x1.cuda()
                batch_y1 = batch_y1.cuda()
                batch_x2 = batch_x2.cuda()
                batch_y2 = batch_y2.cuda()
            else:
                batch_x1 = Variable(batch_x1)
                batch_y1 = Variable(batch_y1)
                batch_x2 = Variable(batch_x2)
                batch_y2 = Variable(batch_y2)
            i = i + 1
            encoded1, decoded1 = SAEncoder(batch_x1.float())
            encoded2, decoded2 = SAEncoder(batch_x2.float())
            if USE_SPARSE:
                rho_hat1 = torch.sum(encoded1, dim=0, keepdim=True)
                rho_hat2 = torch.sum(encoded2, dim=0, keepdim=True)
                # 希望每个样本对应的平均激活都接近于rho，其中rho是我们预定义的一个稀疏指标
                sparsity_penalty1 = params['Beta'] * kl_divergence(params['rho'], rho_hat1)
                sparsity_penalty2 = params['Beta'] * kl_divergence(params['rho'], rho_hat2)
                loss = loss_mse(decoded1.float(), batch_x1.float()) + loss_mse(decoded2.float(), batch_x2.float())+sparsity_penalty+sparsity_penalty2
            else:
                loss = loss_mse(decoded1.float(), batch_x1.float()) + loss_mse(decoded2.float(), batch_x2.float())
            print_loss = loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_loss.append(print_loss)
    plt.plot(y_loss)
    plt.show()

    SAEncoder.eval()

    joblib.dump(sae_model, params['saved'])

except Exception as e:
    print(e)






