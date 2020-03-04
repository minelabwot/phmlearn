#数据读取
import pandas as pd
df1 = pd.read_csv('/datacleaned/train01.csv')
df2 = pd.read_csv('/datacleaned/train02.csv')
df3 = pd.read_csv('/datacleaned/train03.csv')
df4 = pd.read_csv('/datacleaned/train04.csv')
df5 = pd.read_csv('/datacleaned/train05.csv')
df6 = pd.read_csv('/datacleaned/train06.csv')
df7 = pd.read_csv('/datacleaned/train07.csv')
df8 = pd.read_csv('/datacleaned/train08.csv')
df9 = pd.read_csv('/datacleaned/train09.csv')
df10 = pd.read_csv('/datacleaned/train10.csv')
#故障样本整合
df_fault = pd.concat([df1[df1.iloc[:,-2]=='M1a'],df2[df2.iloc[:,-2]=='M2a'],df7[df7.iloc[:,-2]=='M7a'],df9[df9.iloc[:,-2]=='M9a'],df10[df10.iloc[:,-2]=='M10a']])
df_fault = df_fault.reset_index(drop=True)
print(df_fault.shape)
df_fault['label'] = 1
#正常样本整合
df_normal = pd.concat([df3,df4,df6,df8])
df_normal = df_normal.reset_index(drop=True)
print(df_normal.shape)
df_normal['label'] = 0
df_fault.to_csv('/datacsv/data_fault.csv',index=False)
df_normal.to_csv('/datacsv/data_normal.csv',index=False)