# -*- coding: UTF-8 -*-
#简单的求取分帧后假设对应的plc同样分帧，求取主轴负载的均值，最终会输出一个数据集主轴负载的所有分帧平均值和标准差

import pandas as pd
import os

def filesplit(pathname,filename,targetfile):
    #pathname--sensor文件夹绝对路径 filename--对应的plc文件绝对路径 targetfile--目标文件绝对路径
    df = pd.read_csv(filename)
    filelist = os.listdir(pathname)
    filelist.sort(key= lambda x:int(x[:-4]))
    j = 1
    spinload_list = []
    for somefile in filelist:
        filepath = os.path.join(pathname,somefile)
        dk = pd.read_csv(filepath)
        length_window = 12800 #train01 train02 12800 train03 9600 这里处理与sensor分帧相同
        count_window = int(dk.shape[0]/length_window)
        dh = df[df.csv_no==j].reset_index(drop=True)
        baseline = dh.shape[0]//count_window
        print(count_window)
        for i in range(0,count_window):            
            dl = dh.iloc[i*baseline:(i+1)*baseline,:]
            spindle_mean = dl['spindle_load'].mean()
            spindle_std = dl['spindle_load'].std()
            spinload_list.append([spindle_mean,spindle_std])
        j = j + 1
    result_list = pd.DataFrame(spinload_list)
    result_list.to_csv(targetfile,index=None,header=None)

filesplit('/data/01/sensor','/data/01/plc.csv','/feature/plc01.csv')
filesplit('/data/02/sensor','/data/02/plc.csv','/feature/plc02.csv')
#filesplit('/data/03/sensor','/data/03/plc.csv','/feature/plc03.csv')


            

