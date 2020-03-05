# -*- coding: UTF-8 -*-
# 采取数据分帧，将源文件分成长度均为12800的多个子文件
# 如果恰好整除，则尾部可能会多出一个空文件，因此增加了一个尝试性的if用来判断最后一个dataframe是否为空，不过没有实际使用

import os
import pandas as pd


def filesplit(filename, inum, targetfile):
    # filename--文件名，由外部生成传入 inum--外面传入的计数器，代表分出来的帧来源的原始文件 targetfile--目标文件夹的绝对路径
    df = pd.read_csv(filename)
    length_window = 12800  # 每一帧的行数，train01 train02值为12800 train03值为9600 需要修改
    count_window = int(df.shape[0] / length_window)
    print(df.shape[0])
    print(length_window)
    for i in range(0, count_window):
        dk = df.iloc[i * length_window:(i + 1) * length_window, :]
        dk.columns = df.columns
        filenum = i + 1
        dk.to_csv(targetfile + str(inum) + '_' + str(filenum) + '.csv',
                  index=False)
    #dleft = df.iloc[(count_window) * length_window:, :]
    #有时，帧无法按照length_window被整除，因此有如下代码，判断剩下的最后一帧是否为空，若不为空也算为一帧，但有时这一帧由于长度不够，特征有异常，因此最好不计入在内。
    #if dleft.empty is False:
    #    dleft.columns = df.columns
    #    dleft.to_csv(targetfile + str(inum) + '_' + str(count_window + 1) +
    #                 '.csv',
    #                 index=False)


def fileread(pathname, targetfile):
    # 本函数自动调用filesplit函数，pathname--需要分帧的所有文件所在文件夹的绝对路径 targetfile--目标文件夹的绝对路径
    # r'D:\project\PHM2018\train\01-TrainingData-qLua\01\Sensor'
    filelist = os.listdir(pathname)
    filelist.sort(key=lambda x: int(x[:-4]))
    for file in filelist:
        inum = file.split('.')[0]
        filename = os.path.join(pathname, file)
        print(filename, inum)
        filesplit(filename, inum, targetfile)


#fileread('/data/01/sensor','/frame/01_new/')  #12800
#fileread('/data/02/sensor','/frame/02_new/')  #12800
#fileread('/data/03/sensor','/frame/03_new/')  #9600

