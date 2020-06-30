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

class Result:
    explained_variance_score = 0
    mean_absolute_error = 0
    mean_squared_error = 0
    median_absolute_error =0
    r2_score=0
    
params = {}
params['train'] = '/usr/train.csv'
params['test'] = '/usr/test.csv'
params['saved'] = '模型保存路径'

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

    # params['model']为预训练SAE的模型路径
    sae_model = joblib.load(params['saved'])
    sae_weight = sae_model.state_dict()  # sae权重


    class SAE_LSTM(nn.Module):
        def __init__(self):
            super(SAE_LSTM, self).__init__()
            # self.input = nn.Linear(28,128)
            self.encoder = nn.Sequential(
                nn.Linear(166, 166),
                # nn.BatchNorm1d(96),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(166, 166),
                nn.ReLU(),
            )
            self.rnn = nn.LSTM(
                input_size=166,
                hidden_size=166,
                num_layers=1,
                # batch_first=False  # (time_step,batch,input)
                batch_first=False,  # (batch,time_step,input)
                bidirectional=False,
            )
            self.out = nn.Sequential(
                nn.Linear(166, 83),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(83, 1),
                nn.ReLU(),
            )

        def window(self, dat):
            window = 10
            win = Variable(torch.zeros(dat.shape[0] - window + 1, window, dat.shape[1])).cuda()
            rn = dat.shape[0] - window + 1
            for i in range(rn):
                win[i] = dat[i:i + window, :]
            return win

        def forward(self, x, y):
            encode1 = self.encoder(x)  # 512*64
            encode2 = self.encoder(y)  # 512*64
            winx = self.window(encode1)
            r_out, (h_n, h_c) = self.rnn(winx, None)
            # r_out, (h_n, h_c) = self.rnn(x, None)
            out = self.out(r_out[:, -1, :])
            return out, encode1, encode2


    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
        # 将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # 调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)  # /len(kernel_val)


    def loss_mmd(source, target):
        kernel_mul = 2.0
        kernel_num = 5
        fix_sigma = None
        batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        # 根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        # loss = torch.mean(XX + YY - XY -YX) #适用于源域和目标域输入样本数一样
        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        # loss = torch.abs(result)
        return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

    # sae-lstm 加载预训练sae权重
    model1 = SAE_LSTM()
    dtl_params = model1.state_dict()
    pretrained_dict = sae_weight
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
                       dtl_params}
    dtl_params.update(pretrained_dict)
    model1.load_state_dict(dtl_params)

    for epoch in range(params['iterations']):
        i = 0
        for data1, data3 in zip(train1_loader, train2_loader):
            i = i + 1
            batch_x1, batch_y1 = data1
            batch_x3, batch_y3 = data3
            if torch.cuda.is_available():
                batch_x1 = batch_x1.cuda()
                batch_y1 = batch_y1.cuda()
                batch_x3 = batch_x3.cuda()
                batch_y3 = batch_y3.cuda()
            else:
                batch_x1 = Variable(batch_x1)
                batch_y1 = Variable(batch_y1)
                batch_x3 = Variable(batch_x3)
                batch_y3 = Variable(batch_y3)
            predict1, encode1, encode3 = model1(batch_x1.float(), batch_x3.float())
            loss = criterion(predict1, batch_y1[9:, :].float()) + 0.8 * loss_mmd(encode1, encode3)
            print_loss = loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_loss.append(print_loss)

    # 训练完之后输出评价结果

    res = {}

    res['varianceScore'] = explained_variance_score(yt / 60, yp / 60, multioutput="uniform_average")
    res['absoluteError'] = mean_absolute_error(yt / 60, yp / 60, multioutput="uniform_average")
    res['squaredError'] = np.sqrt(mean_squared_error(yt / 60, yp / 60, multioutput="uniform_average"))
    res['medianSquaredError'] = median_absolute_error(yt / 60, yp / 60)
    res['r2Score'] = r2_score(yt / 60, yp / 60, multioutput="uniform_average")

    print(json.dumps(res))

except Exception as e:
    print(e)






