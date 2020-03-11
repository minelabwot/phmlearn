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

columns_name = ['sensor_ave','sensor_std','sensor_var','sensor_skew','sensor_peak','sensor_ptp','sensor_ratio',
                'sensor_ca5e','sensor_cd5e','sensor_cd4e','sensor_cd3e','sensor_cd2e','sensor_cd1e','sensor_ar_1',
                'sensor_ar_2','sensor_ar_3','sensor_ar_4','sensor_ar_5','sensor_ar_6','sensor_ar_7','sensor_ar_8',
                'sensor_ar_9','sensor_ar_10','sensor_ar_11','sensor_ar_12','sensor_ar_13','sensor_ar_14','sensor_ar_15',
                'sensor_ar_16','sensor_ar_17','sensor_ar_18','sensor_ar_19','sensor_ar_20','sensor_ar_21','sensor_ar_22',
                'sensor_ar_23','sensor_ar_24','sensor_ar_25','sensor_ar_26','sensor_ar_27','sensor_ar_28','sensor_ar_29',
                'sensor_ar_30','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10','mfcc_11',
                'mfcc_12','mfcc_13']
column_vib1 = [ x + '_vib1' for x in columns_name]
column_vib2 = [ x + '_vib2' for x in columns_name]
column_curr = [ x + '_curr' for x in columns_name]
column_all = column_vib1 + column_vib2 + column_curr

def base_feaget(root_file):
    df = pd.read_csv(root_file)
    result_list = []
    for i in df.columns:
        df_i = np.asarray(df[i])
        flist, plist = signal.welch(df_i, 25600)
        main_ener1 = np.square(plist[np.logical_and(flist >= 1600,
                                                    flist < 2400)]).sum()
        main_ener2 = np.square(plist[np.logical_and(flist >= 3600,
                                                    flist < 3950)]).sum()
        ratio = main_ener1 / main_ener2
        # wave
        cA5, cD5, cD4, cD3, cD2, cD1 = wavedec(df_i, 'db10', level=5)
        ener_cA5 = np.square(cA5).sum()
        ener_cD5 = np.square(cD5).sum()
        ener_cD4 = np.square(cD4).sum()
        ener_cD3 = np.square(cD3).sum()
        ener_cD2 = np.square(cD2).sum()
        ener_cD1 = np.square(cD1).sum()
        # ar
        method1_maxlag = 29
        model_ar = sm.tsa.AR(df_i)
        ar_result = model_ar.fit(method1_maxlag)
        # mfcc
        mfccs = python_speech_features.mfcc(df_i,
                                            samplerate=25600,
                                            winlen=0.5,
                                            winstep=0.1,
                                            nfft=12800)
        # output
        # 一些基本特征
        list_para = [
            df_i.mean(), df_i.std(),
            np.var(df_i),
            stats.skew(df_i),
            stats.kurtosis(df_i), df_i.ptp(), ratio, ener_cA5, ener_cD5,
            ener_cD4, ener_cD3, ener_cD2, ener_cD1
        ]
        list_para.extend(ar_result.params)
        list_para.extend(mfccs[0][1:])
        result_list.extend(list_para)
    return result_list

def fileread(pathname, plcfile, targetname):
    #pathname--分帧后sensor文件夹绝对路径 plcfile--对应的plc文件绝对路径 targetfile--目标文件绝对路径
    # r'D:\project\PHM2018\train\01-TrainingData-qLua\01\Sensor'
    #sensor-processing
    print("now processing sensor")
    filelist = os.listdir(pathname)
    filelistb = sorted(filelist,
                       key=lambda x:
                       (int(x.split('_')[0]), int(x.split('_')[1][:-4])))
    filewr = []
    whattowr = []
    for somefile in filelistb:
        filename = os.path.join(pathname, somefile)
        try:
            temp_h = base_feaget(filename)
            whattowr.append(temp_h)
        except:
            whattowr.append([])
        print(filename)
        filewr.append(somefile.split('.')[0])
    wrtocsv = pd.DataFrame(whattowr,columns=column_all)
    #plc-processing
    print("now processing plc")
    spinload_list = []
    df_plc = pd.read_csv(plcfile)
    orifile = sorted(list(set([x.split('_')[0] for x in filelistb])))
    for i in orifile:
        count_window = 0
        dh = df_plc[df_plc['csv_no']==int(i)].reset_index(drop=True)
        for somefile in filelistb:
            matchchar = i + '_'
            if somefile.startswith(matchchar):
                count_window += 1
        baseline = dh.shape[0]//count_window
        print(count_window)
        for j in range(0,count_window):            
            dl = dh.iloc[j*baseline:(j+1)*baseline,:]
            spindle_mean = dl['spindle_load'].mean()
            spinload_list.append(spindle_mean)
    wrtocsv['spindle_load'] = spinload_list
    wrtocsv['file_source'] = filewr
    wrtocsv.to_csv(targetname, index=False)

if __name__ == '__main__':
    fileread('/frame/01_new/','/data/01/plc.csv','/feature/methodfea_train01.csv')
    fileread('/frame/02_new/','/data/02/plc.csv','/feature/methodfea_train02.csv')
    fileread('/frame/03_new/','/data/03/plc.csv','/feature/methodfea_train03.csv') 

