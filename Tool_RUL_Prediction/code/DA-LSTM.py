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
BATCH_SIZE = 512

torch_x_train1 = torch.from_numpy(X1_scaler)
torch_y_train1 = torch.from_numpy(Y1.values)
train1_dataset = Data.TensorDataset(torch_x_train1,torch_y_train1)
train1_loader = DataLoader(train1_dataset, batch_size=550, shuffle=False)

torch_x_train2 = torch.from_numpy(X2_scaler)
torch_y_train2 = torch.from_numpy(Y2.values)
train2_dataset = Data.TensorDataset(torch_x_train2, torch_y_train2)
train2_loader = DataLoader(train2_dataset, batch_size=550, shuffle=False)

torch_x_train3 = torch.from_numpy(X3_scaler)
torch_y_train3 = torch.from_numpy(Y3.values)
train3_dataset = Data.TensorDataset(torch_x_train3, torch_y_train3)
train3_loader = DataLoader(train3_dataset, batch_size=512, shuffle=False)

#自编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
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
#初始化自编码器权重
def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(6.0/(fan_in + fan_out))
        #m.weight.data.normal_(0.0, variance)
        m.weight.data.uniform_(-variance, variance)
LR = 0.005
AEncoder = AutoEncoder()
AEncoder.apply(weight_init)
if torch.cuda.is_available():
    AEncoder = AEncoder.cuda()
optimizer = torch.optim.Adam(AEncoder.parameters(), lr=LR,weight_decay=1e-8)
loss_mse = nn.MSELoss()

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
        encoded1, decoded1 = AEncoder(batch_x1.float())
        encoded2, decoded2 = AEncoder(batch_x2.float())
        loss = loss_mse(decoded1.float(), batch_x1.float()) + loss_mse(decoded2.float(), batch_x2.float())

        print_loss = loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_loss.append(print_loss)
plt.plot(y_loss)
plt.show()

AEncoder.eval()
#模型参数
auto = AEncoder.state_dict()

#DA-LSTM迁移学习模型
class DTL(nn.Module):
    def __init__(self):
        super(DTL, self).__init__()
        #self.input = nn.Linear(28,128)
        self.encoder = nn.Sequential(
            nn.Linear(166, 166),
            #nn.BatchNorm1d(96),
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
    def window(self, dat):
        window = 10
        win = Variable(torch.zeros(dat.shape[0]-window+1,window,dat.shape[1])).cuda()
        rn = dat.shape[0]-window+1
        for i in range(rn):
            win[i] = dat[i:i+window,:]
        return win
    def forward(self,x,y):
        encode1 = self.encoder(x)  #512*64
        encode2 = self.encoder(y)  #512*64
        winx = self.window(encode1)
        r_out, (h_n, h_c) = self.rnn(winx, None)
        #r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:]) #切片-1代表倒数第一索引，即取lstm 最后一个cell的输出
        return out,encode1,encode2

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)
def loss_mmd(source, target):
    kernel_mul=2.0
    kernel_num=5
    fix_sigma=None
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    #loss = torch.mean(XX + YY - XY -YX) #适用于源域和目标域输入样本数一样
    loss = torch.mean(XX)+torch.mean(YY)-2*torch.mean(XY)
    #loss = torch.abs(result)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

model2 = DTL()
#加载预训练编码器的参数
dtl_params = model2.state_dict()
pretrained_dict = auto
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in dtl_params}
dtl_params.update(pretrained_dict)
model2.load_state_dict(dtl_params)
#定义超参数
if torch.cuda.is_available():
    model2 = model2.cuda()
LR = 0.003
optimizer = torch.optim.Adam(model2.parameters(), lr=LR,weight_decay=1e-8) #优化器
criterion = nn.MSELoss()
y_loss = []

for epoch in range(150):
    i = 0
    for data1,data3 in zip(train1_loader,train2_loader):
        i=i+1
        batch_x1,batch_y1 = data1
        batch_x3,batch_y3 = data3
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
        predict1,encode1,encode3 = model2(batch_x1.float(),batch_x3.float())
        loss = criterion(predict1, batch_y1[9:,:].float())+0.8*loss_mmd(encode1,encode3)
        print_loss = loss.data.item()
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    y_loss.append(print_loss)
plt.plot(y_loss)
plt.show()

model2.eval()

out1,enco1,enco2 = model2(torch_x_train2.cuda().float(),torch_x_train2.cuda().float())
yp = out1.data.cpu().numpy()
yt = Y2[9:].values

#评价结果
print("RMSE is:")
print(np.sqrt(mean_squared_error(yt/60,yp/60)))




