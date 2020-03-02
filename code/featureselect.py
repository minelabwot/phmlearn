import pandas as pd
feature_normal = pd.read_csv('/featureset/feature_normal.csv')
feature_fault = pd.read_csv('/featureset/feature_fault.csv')
#这里将正常样本和故障征兆样本合并
feature_final = pd.concat([feature_normal,feature_fault])
#重置索引，否则可能造成索引混乱
feature_final = feature_final.reset_index(drop=True)
#可以修改以下list调整需要留下的特征
feature_selected_list = ['CEX_1XRatio','CEX_2XRatio','CEX_3XRatio',
                         'CEY_1XRatio','CEY_2XRatio','CEY_3XRatio',
                         'NCEX_1XRatio','NCEX_2XRatio','NCEX_3XRatio',
                         'NCEY_1XRatio','NCEY_2XRatio','NCEY_3XRatio',
                         'SDA_1XRatio','SDA_2XRatio','SDA_3XRatio',
                         'SDB_1XRatio','SDB_2XRatio','SDB_3XRatio',
                         'CEX_time_ptp','CEY_time_ptp','NCEX_time_ptp','NCEY_time_ptp',
                         'SDA_time_ptp','SDB_time_ptp','label']
feature_selected = feature_final[feature_selected_list]
#筛选后特征保存
feature_selected.to_csv('/featureset/feature_selected.csv',index=False)

#用for循环同样处理测试集
for i in range(11,19):
    feature_test = pd.read_csv('/featureset/feature_test' + str(i) +'.csv')
    feature_selected_test = feature_test[feature_selected_list]
    feature_selected_test.to_csv('//featureset/test'+ str(i) + '_selected.csv',index=False)