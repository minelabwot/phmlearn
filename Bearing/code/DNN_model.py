import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error  # 均方误差
import sys
import json
import json
import traceback
import warnings
warnings.filterwarnings('ignore')

params = {}
params['traindata_path'] = ''
params['testdata_path'] = ''
params['epoch_num'] = 200 #模型迭代次数
params['LR'] = 0.001

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


    train_data = pd.DataFrame(pd.read_csv(params['traindata_path'] ))
    test_data = pd.DataFrame(pd.read_csv(params['testdata_path']))

    X1 = train_data.drop(['Label'], axis=1)
    X7 = test_data.drop(['Label'], axis=1)

    Y1 = train_data.loc[:, ['Label']]
    Y7 = test_data.loc[:, ['Label']]


    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    X1_scaler =  X_scaler.fit_transform(X1)
    X7_scaler =  X_scaler.fit_transform(X7)

    BATCH_SIZE = 512

    torch_x_train1 = torch.from_numpy(X1_scaler)
    torch_y_train1 = torch.from_numpy(Y1.values)
    train1_dataset = Data.TensorDataset(torch_x_train1, torch_y_train1)
    train1_loader = DataLoader(train1_dataset, batch_size=512, shuffle=False)

    torch_x_train7 = torch.from_numpy(X7_scaler)
    torch_y_train7 = torch.from_numpy(Y7.values)
    train7_dataset = Data.TensorDataset(torch_x_train7, torch_y_train7)
    train7_loader = DataLoader(train7_dataset, batch_size=512, shuffle=False)


    class DNN(nn.Module):
        def __init__(self):
            super(DNN, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(90, 90),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(90, 90),
                nn.ReLU(),
                nn.Linear(90, 45),
                nn.ReLU(),
                nn.Linear(45, 1),
                nn.ReLU(),
            )

        def forward(self, x):
            output = self.layers(x)
            return output


    model1 = DNN()
    if torch.cuda.is_available():
        model1 = model1.cuda()
    optimizer = torch.optim.Adam(model1.parameters(), lr=params['LR'], weight_decay=1e-8)  
    criterion1 = nn.MSELoss()  # 损失函数
    y_loss = []
    for epoch in range(params['epoch_num']):
        for data1 in train1_loader:

            batch_x1, batch_y1 = data1
            if torch.cuda.is_available():
                batch_x1 = batch_x1.cuda()
                batch_y1 = batch_y1.cuda()
            else:
                batch_x1 = batch_x1.cuda()
                batch_y1 = batch_y1.cuda()

            predict1 = model1(batch_x1.float())

            loss = criterion1(predict1, batch_y1.float())
            print_loss = loss.data.item()
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()  
        y_loss.append(print_loss)
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    model1.eval()

    torch_x_train7 = torch_x_train7.cuda()
    out1 = model1(torch_x_train7.float())
    yp = out1.cpu().detach().numpy()
    yt = Y7.values



    res = {}
    res['rmse'] = np.sqrt(mean_squared_error(yt/60,yp/60))

    print(json.dumps(res))
except Exception as e:
    traceback.print_exc()





