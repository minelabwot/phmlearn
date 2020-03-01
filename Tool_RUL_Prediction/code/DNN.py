import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error #均方误差
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

#读取数据
data1 = pd.DataFrame(pd.read_csv('/home/FBH/Tool_Data/methodfea_train01.csv'))
data2 = pd.DataFrame(pd.read_csv('/home/FBH/Tool_Data/methodfea_train02.csv'))
data3 = pd.DataFrame(pd.read_csv('/home/FBH/Tool_Data/methodfea_train03.csv'))

X1 = data1.drop(['label'], axis=1)
X2 = data2.drop(['label'], axis=1)
X3 = data3.drop(['label'], axis=1)

Y1 = data1.loc[:,['label']]
Y2 = data2.loc[:,['label']]
Y3 = data3.loc[:,['label']]

#归一化
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X1_scaler =  X_scaler.fit_transform(X1)
X2_scaler =  X_scaler.fit_transform(X2)
X3_scaler =  X_scaler.fit_transform(X3)

#转换成张量格式
BATCH_SIZE = 512

torch_x_train1 = torch.from_numpy(X1_scaler)
torch_y_train1 = torch.from_numpy(Y1.values)
train1_dataset = Data.TensorDataset(torch_x_train1,torch_y_train1)
train1_loader = DataLoader(train1_dataset, batch_size=512, shuffle=False)

torch_x_train2 = torch.from_numpy(X2_scaler)
torch_y_train2 = torch.from_numpy(Y2.values)
train2_dataset = Data.TensorDataset(torch_x_train2, torch_y_train2)
train2_loader = DataLoader(train2_dataset, batch_size=512, shuffle=False)

torch_x_train3 = torch.from_numpy(X3_scaler[:,:])
torch_y_train3 = torch.from_numpy(Y3[:].values)
train3_dataset = Data.TensorDataset(torch_x_train3, torch_y_train3)
train3_loader = DataLoader(train3_dataset, batch_size=512, shuffle=False)

#模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(166, 166),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(166, 166),
            nn.ReLU(),
            nn.Linear(166, 83),
            nn.ReLU(),
            nn.Linear(83, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.layers(x)
        return output

model1 = DNN()
if torch.cuda.is_available():
    model1 = model1.cuda()
LR = 0.001  #学习率
optimizer = torch.optim.Adam(model1.parameters(), lr=LR,weight_decay=1e-8) #优化器
criterion1 = nn.MSELoss() #损失函数
rounds = 200   #模型迭代次数

y_loss = []

for epoch in range(rounds):
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
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    y_loss.append(print_loss)
    # print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
plt.plot(y_loss)
plt.show()

model1.eval()

torch_x_train2 = torch_x_train2.cuda()#刀具2为测试集
out1=model1(torch_x_train2.float())
yp = out1.cpu().detach().numpy()
yt = Y2.values

#评价结果
print("RMSE is:")
print(np.sqrt(mean_squared_error(yt/60,yp/60)))




