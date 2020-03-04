from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import pandas as pd

def train():
    #模型初始化，设置random_state保证可复现性，便于观察优化
    train_data = pd.read_csv('/featureset/feature_selected.csv')
    train_data_y = train_data['label']
    #除去标签的所有列就是特征
    train_data_x = train_data.drop(['label'],axis=1)
    model_neigh_default = KNeighborsClassifier(n_neighbors=5)
    #模型训练
    model_neigh_default.fit(train_data_x,train_data_y)
    joblib.dump(model_neigh_default,'/model/model_KNN.pkl')

#这里首先定义judge函数，以threhold为阈值，根据模型给出的分类概率进行判断，大于 
#threhold为正常样本，反之为故障征兆样本
def judge(input_pred,threshold):
    return_pred = list(np.zeros(len(input_pred)))
    for i in range(0,len(input_pred)):
        if (input_pred[i][0]>threshold):
            return_pred[i] = 0
        else:
            return_pred[i] = 1
    return return_pred

def test_knn():
    #加载模型
    model = joblib.load('/model/model_KNN.pkl')
    #读取测试集,来源的特征文件同样经过特征提取和选择，只是不加标签，'label'列中表示
    #其机组和数据组的信息，例如'M11_1'，方便我们统计各数据组的分类概率统计值进行排序
    test_m11 = pd.read_csv('/featureset/test11_selected.csv')
    test_m12 = pd.read_csv('/featureset/test12_selected.csv')
    test_m13 = pd.read_csv('/featureset/test13_selected.csv')
    test_m14 = pd.read_csv('/featureset/test14_selected.csv')
    test_m15 = pd.read_csv('/featureset/test15_selected.csv')
    test_m16 = pd.read_csv('/featureset/test16_selected.csv')
    test_m17 = pd.read_csv('/featureset/test17_selected.csv')
    test_m18 = pd.read_csv('/featureset/test18_selected.csv')
    #对8个测试集循环进行测试
    h = 11
    for test_i in [test_m11,test_m12,test_m13,test_m14,test_m15,test_m16,
                   test_m17,test_m18]:
        test_name = 'test_m' + str(h)
        print('The result of test is:')
        print('filename  datagroup  normal  fault  fault-probability-of-datagroup')
        max_j = {}
        for j in range(1,6):
            labelname = 'M' + str(h) + '_' + str(j)
            test_j = test_i[test_i['label']==labelname]
            test_feature = test_j.drop(['label'],axis=1)
            y_pred = model.predict_proba(test_feature)
            y_pred_binary = judge(y_pred,0.5)
            y_pred_mean = np.mean([x[1] for x in y_pred])
            max_j[j] = y_pred_mean
            print(test_name,j,y_pred_binary.count(0),y_pred_binary.count(1),
                  y_pred_mean)
        h = h + 1
        print("The order a-e is:")
        for k in sorted(max_j,key=max_j.__getitem__,reverse=True):
            print(k,max_j[k])

train()
test_knn()