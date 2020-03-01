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

#数据格式
#滑窗，处理成3d数据
def window_data(dat):
    window = 10
    win = np.zeros((dat.shape[0]-window+1, window, dat.shape[1]))
    rn = dat.shape[0]-window+1
    for i in range(rn):
        win[i] = dat[i:i+window,:]
    return win

window_x_train1 = torch.from_numpy(window_data(X1_scaler))
window_x_train2 = torch.from_numpy(window_data(X2_scaler))
window_x_train3 = torch.from_numpy(window_data(X3_scaler))

window_y_train1 = torch.from_numpy(Y1[9:].values) #window-1
window_y_train2 = torch.from_numpy(Y2[9:].values)
window_y_train3 = torch.from_numpy(Y3[9:].values)

BATCH_SIZE = 512
train1_dataset = Data.TensorDataset(window_x_train1,window_y_train1)
train2_dataset = Data.TensorDataset(window_x_train2,window_y_train2)
train3_dataset = Data.TensorDataset(window_x_train3,window_y_train3)

train1_loader = DataLoader(train1_dataset, batch_size=BATCH_SIZE, shuffle=False)
train2_loader = DataLoader(train2_dataset, batch_size=BATCH_SIZE, shuffle=False)
train3_loader = DataLoader(train3_dataset, batch_size=BATCH_SIZE, shuffle=False)

#模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=166,
            hidden_size=166,
            num_layers=1,
            batch_first = False,   # (batch,time_step,input)
            bidirectional=False,
        )
        self.out = nn.Sequential(
            nn.Linear(166,83),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(83, 1),
            nn.ReLU(),
        )
    def forward(self,x):
        r_out1, (h_n1, h_c1) = self.rnn(x, None)
        #encode1 = r_out1[:,-1,:]
        out = self.out(r_out1[:,-1,:]) #切片-1代表倒数第一索引，即取lstm 最后一个cell的输出
        return out

model2 = DNN()
if torch.cuda.is_available():
    model2 = model2.cuda()
LR = 0.01  #学习率
optimizer = torch.optim.Adam(model2.parameters(), lr=LR,weight_decay=1e-8) #优化器
criterion1 = nn.MSELoss() #损失函数
rounds = 200   #模型迭代次数
model2.train()
y_loss = []

for epoch in range(200):
    i = 0
    for data1 in train1_loader:
        i=i+1
        batch_x1,batch_y1 = data1
        if torch.cuda.is_available():
            batch_x1 = batch_x1.cuda()
            batch_y1 = batch_y1.cuda()
        else:
            batch_x1 = Variable(batch_x1)
            batch_y1 = Variable(batch_y1)
        predict1 = model2(batch_x1.float())
        loss = criterion(predict1, batch_y1[:,:].float())
        print_loss = loss.data.item()
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    y_loss.append(print_loss)
plt.plot(y_loss)
plt.show()

model2.eval()

out1 = model2(window_x_train2.cuda().float())
yp = out1.data.cpu().numpy()
yt = Y2[9:].values

#评价结果
print("RMSE is:")
print(np.sqrt(mean_squared_error(yt/60,yp/60)))




