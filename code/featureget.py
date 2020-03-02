#导入所需库函数
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy import signal
#列名定义
df_out_columns = ['time_mean','time_std','time_max','time_min','time_rms',
                  'time_ptp','time_median','time_iqr','time_pr',
                  'time_skew','time_kurtosis','time_var','time_amp',
                  'time_smr','time_wavefactor','time_peakfactor',
                  'time_pulse','time_margin','1X','2X','3X','1XRatio',
                  '2XRatio','3XRatio']
CEX_columns = ['CEX_' + i for i in df_out_columns]
CEY_columns = ['CEY_' + i for i in df_out_columns]
NCEX_columns = ['NCEX_' + i for i in df_out_columns]
NCEY_columns = ['NCEY_' + i for i in df_out_columns]
SDA_columns = ['SDA_' + i for i in df_out_columns]
SDB_columns = ['SDB_' + i for i in df_out_columns]
label_columns = ['label']
full_columns = CEX_columns + CEY_columns + NCEX_columns + NCEY_columns + SDA_columns + SDB_columns + label_columns

def featureget(df_line):
    #提取时域特征
    time_mean = df_line.mean()
    time_std = df_line.std()
    time_max = df_line.max()
    time_min = df_line.min()
    time_rms = np.sqrt(np.square(df_line).mean().astype(np.float64))
    time_ptp = np.asarray(df_line).ptp()
    time_median = np.median(df_line)
    time_iqr = np.percentile(df_line,75)-np.percentile(df_line,25)
    time_pr = np.percentile(df_line,90)-np.percentile(df_line,10)
    time_skew = stats.skew(df_line)
    time_kurtosis = stats.kurtosis(df_line)
    time_var = np.var(df_line)
    time_amp = np.abs(df_line).mean()
    time_smr = np.square(np.sqrt(np.abs(df_line).astype(np.float64)).mean())
    #下面四个特征需要注意分母为0或接近0问题，可能会发生报错
    time_wavefactor = time_rms/time_amp
    time_peakfactor = time_max/time_rms
    time_pulse = time_max/time_amp
    time_margin = time_max/time_smr
    #提取频域特征倍频能量以及能量占比
    plist_raw = np.fft.fft(list(df_line), n=1024)
    plist = np.abs(plist_raw)
    plist_energy = (np.square(plist)).sum()
    #在傅里叶变换结果中，在32点处的幅值为一倍频幅值，64点处幅值为二倍频幅值，96点处为三倍频幅值，因此提取这三处幅值并计算能量占比
    return_list = [
    time_mean,time_std,time_max,time_min,time_rms,time_ptp, 
    time_median,time_iqr,time_pr,time_skew,time_kurtosis,
    time_var,time_amp,time_smr,time_wavefactor,time_peakfactor,
    time_pulse,time_margin,plist[32], plist[64], plist[96],
    np.square(plist[32]) / plist_energy,
    np.square(plist[64]) / plist_energy,
    np.square(plist[96]) / plist_energy
]
    return return_list

df_normal = pd.read_csv('/datacsv/data_normal.csv')
feature_normal = []
for i in range(0,df_normal.shape[0]):
    feature_line = []
    print(i)
    #对整合后数据每个传感器采集的波形循环提取特征
    for j in range(0,6):
        feature_line.extend(featureget(df_normal.iloc[i,2+1026*j:1026*(j+1)]))
    feature_line.append(df_normal['label'][i])
    feature_normal.append(feature_line)
#输出到特定文件中
feature_normal = pd.DataFrame(feature_normal,columns=full_columns)
feature_normal.to_csv('/featureset/feature_normal.csv',index=False)

df_fault = pd.read_csv('/datacsv/data_fault.csv')
feature_fault = []
for i in range(0,df_fault.shape[0]):
    feature_line = []
    print(i)
    #对整合后数据每个传感器采集的波形循环提取特征
    for j in range(0,6):
        feature_line.extend(featureget(df_fault.iloc[i,2+1026*j:1026*(j+1)]))
    feature_line.append(df_fault['label'][i])
    feature_fault.append(feature_line)
#输出到特定文件中
feature_fault = pd.DataFrame(feature_fault,columns=full_columns)
feature_fault.to_csv('/featureset/feature_fault.csv',index=False)

#使用for循环对测试集同样提取特征
for k in range(11,19):
    df_test = pd.read_csv('/datacsv/test' + str(k) +'.csv')
    feature_test = []
    for i in range(0,df_test.shape[0]):
        feature_line = []
        print(i)
        #对整合后数据每个传感器采集的波形循环提取特征
        for j in range(0,6):
            feature_line.extend(featureget(df_test.iloc[i,2+1026*j:1026*(j+1)]))
        feature_line.append(df_test.iloc[i,-2])
        feature_test.append(feature_line)
    #输出到特定文件中
    feature_test = pd.DataFrame(feature_test,columns=full_columns)
    feature_test.to_csv('/featureset/feature_test' + str(k) +'.csv',index=False)
