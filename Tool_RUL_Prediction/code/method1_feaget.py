# -*- coding: UTF-8 -*-
# 提取特征，特征说明见feature说明

import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy import signal
import statsmodels.api as sm
from itertools import chain
from pywt import wavedec
import python_speech_features


def method1_feaget(root_file, name_file):
    df = pd.read_csv(root_file)
    result_list = []
    for i in df.columns:
        flist, plist = signal.welch(df[i], 25600)
        main_ener1 = np.square(plist[np.logical_and(flist >= 1600,
                                                    flist < 2400)]).sum()
        main_ener2 = np.square(plist[np.logical_and(flist >= 3600,
                                                    flist < 3950)]).sum()
        ratio = main_ener1 / main_ener2
        # wave
        cA5, cD5, cD4, cD3, cD2, cD1 = wavedec(df[i], 'db10', level=5)
        ener_cA5 = np.square(cA5).sum()
        ener_cD5 = np.square(cD5).sum()
        ener_cD4 = np.square(cD4).sum()
        ener_cD3 = np.square(cD3).sum()
        ener_cD2 = np.square(cD2).sum()
        ener_cD1 = np.square(cD1).sum()
        # ar
        method1_maxlag = 29
        model_ar = sm.tsa.AR(df[i])
        ar_result = model_ar.fit(method1_maxlag)
        # output
        list_para = [
            df[i].mean(), df[i].std(),
            np.var(df[i]),
            stats.skew(df[i]),
            stats.kurtosis(df[i]), df[i].ptp(), ratio, ener_cA5, ener_cD5,
            ener_cD4, ener_cD3, ener_cD2, ener_cD1
        ]
        #print(len(ar_result.params))
        list_para.extend(ar_result.params)
        result_list.extend(list_para)
    return result_list


def mfcc_feaget(root_file, name_file):
    df = pd.read_csv(root_file)
    result_list = []
    list_name = name_file.split('.')[0]
    for i in df.columns:
        mfccs = python_speech_features.mfcc(np.array(df[i]),
                                            samplerate=25600,
                                            winlen=0.5,
                                            winstep=0.1,
                                            nfft=12800)
        result_list.extend(mfccs[0][1:])
    #下面的代码没测试过，其存在原因在于原始版本的分帧后文件名生成是用'-'，这种格式在excel里会被重写成日期，
    # 用'_'就没有这种影响，之前的frame_divide也已经重写成生成的文件名中有'_'
    if ('-' in list_name):
        list_name = list_name.replace('-', '_')
    result_list.append(list_name)
    return result_list


def fileread(pathname, targetname):
    # r'D:\project\PHM2018\train\01-TrainingData-qLua\01\Sensor'
    filelist = os.listdir(pathname)
    #同样在下面一行读取文件列表时注意，如果利用frame_divide进行了重新分帧，则文件名已经完全被重写成'_'，
    # 此时下面用来分割字符串的'-'需要修改成'_'，如果没重新做过分帧则不必修改
    filelistb = sorted(filelist,
                       key=lambda x:
                       (int(x.split('_')[0]), int(x.split('_')[1][:-4])))
    #filelist.sort(key= lambda x:int(x[:-4]))
    whattowr = []
    for somefile in filelistb:
        filename = os.path.join(pathname, somefile)
        temp_h = method1_feaget(filename, somefile)
        temp_h.extend(mfcc_feaget(filename, somefile))
        whattowr.append(temp_h)
        print(filename)
    wrtocsv = pd.DataFrame(whattowr)
    wrtocsv.to_csv(targetname, index=False, header=False)


fileread('/frame/01_new/','/feature/methodfea_train01.csv')
fileread('/frame/02_new/','/feature/methodfea_train02.csv')
fileread('/frame/03_new/','/feature/methodfea_train03.csv')

