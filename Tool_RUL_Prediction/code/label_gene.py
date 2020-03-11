# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
#为提取特征后的特征文件打标签

#均分标签函数，原刀具数据是五分钟随机取一分钟，这里为了保证标签的连续性，将五分钟的长度均分打给数据行的形式进行打标签。默认使用这个函数。
def label_generator(n,input_line):
    result = []
    for i in range(0,len(input_line)):
        start_va = (n-i)*300.0
        end_va = (n-i-1)*300.0
        line_va = list(np.linspace(end_va,start_va,input_line[i],endpoint=False))
        result.extend(list(reversed(line_va)))
    print(len(result))
    return result

#随机标签函数，忠实于数据说明，随机选取数据所在五分钟内的一分钟打标签。
def label_generator_random(n,group_train):
    length_train = len(group_train)
    print(length_train)
    random_train = np.random.randint(1,6,size=length_train)
    print(random_train)
    all_group = []
    for i in range(0,length_train):
        if (random_train[i] == 5):
            upline = (n-i)*300 - 0.5
            downline = upline - (group_train[i]-1)*0.5
        else:
            downline = (n-i-1)*300 + (random_train[i]-1)*60
            upline = downline + (group_train[i]-1)*0.5
        label_group = np.linspace(upline,downline,group_train[i])
        #print(len(label_group))
        all_group.extend(label_group)
    return all_group

def label_gena(n,filename,targetname,random=False):
    #n为该刀具原始文件数目，48(01/02)或37(03),filename为提取特征后的文件，targetname为输出文件，random为是否采用随机标签法，默认不采用。
    file = pd.read_csv(filename)
    count_train = np.zeros(n)
    for i in file.file_source:
        count_train[int(i.split('_')[0])-1] += 1
    if random:
       label_g = label_generator_random(n,count_train)
    else:
       label_g = label_generator(n,count_train)
    file['label'] = label_g
    file.to_csv(targetname,index=False)

if __name__ == '__main__':
    label_gena(48,'/feature/methodfea_train01.csv','/feature/methodfea_train01_label.csv')
    label_gena(48,'/feature/methodfea_train02.csv','/feature/methodfea_train02_label.csv')
    label_gena(37,'/feature/methodfea_train03.csv','/feature/methodfea_train03_label.csv')
