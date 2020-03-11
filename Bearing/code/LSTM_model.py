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
from sklearn.metrics import mean_squared_error  
import sys
import json
import json
import traceback
import warnings
warnings.filterwarnings('ignore')

params = {}
params['traindata_path'] = ''
params['testdata_path'] = ''
params['epoch_num'] = 200 
params['LR'] = 0.001
params['window_num'] = 10

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


    # 滑窗，处理成3d数据
    def window_data(dat):
        window = params['window_num']
        win = np.zeros((dat.shape[0] - window + 1, window, dat.shape[1]))
        rn = dat.shape[0] - window + 1
        for i in range(rn):
            win[i] = dat[i:i + window, :]
        return win


    window_x_train1 = torch.from_numpy(window_data(X1_scaler))
    window_x_train7 = torch.from_numpy(window_data(X7_scaler))

    window_y_train1 = torch.from_numpy(Y1[params['window_num']-1:].values)  # window-1
    window_y_train7 = torch.from_numpy(Y7[params['window_num']-1:].values)

    BATCH_SIZE = 512
    train1_dataset = Data.TensorDataset(window_x_train1, window_y_train1)
    train7_dataset = Data.TensorDataset(window_x_train7, window_y_train7)

    train1_loader = DataLoader(train1_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train7_loader = DataLoader(train7_dataset, batch_size=BATCH_SIZE, shuffle=False)


    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(
                input_size=90,
                hidden_size=90,
                num_layers=1,
                batch_first=False,  # (batch,time_step,input)
                bidirectional=False,
            )
            self.out = nn.Sequential(
                nn.Linear(90, 45),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(45, 1),
                nn.ReLU(),
            )

        def forward(self, x):
            r_out1, (h_n1, h_c1) = self.rnn(x, None)
            # encode1 = r_out1[:,-1,:]
            out = self.out(r_out1[:, -1, :])  
            return out


    model2 = RNN()
    if torch.cuda.is_available():
        model2 = model2.cuda()
    optimizer = torch.optim.Adam(model2.parameters(), lr=params['LR'], weight_decay=1e-8)  
    criterion = nn.MSELoss()
    # criterion_mmd = nn.L1Loss()
    model2.train()
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

            predict1 = model2(batch_x1.float())

            loss = criterion(predict1, batch_y1.float())
            print_loss = loss.data.item()
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()  
        y_loss.append(print_loss)
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    model2.eval()

    out1 = model2(window_x_train7.cuda().float())
    yp = out1.data.cpu().numpy()
    yt = Y7[params['window_num']-1:].values

    res = {}
    res['rmse'] = np.sqrt(mean_squared_error(yt/60,yp/60))

    print(json.dumps(res))
except Exception as e:
    traceback.print_exc()





