import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#import xgboost as xgb
from matplotlib import pyplot
import lightgbm as lgb
#from xgboost import plot_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error  # 均方误差
import sys
import json
import json
import traceback
import warnings
warnings.filterwarnings('ignore')

params = {}
params['traindata_path'] = ''
params['testdata_path'] = ''

argvs = sys.argv
try:

    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

    data1 = pd.DataFrame(pd.read_csv(params['traindata_path']))
    data7 = pd.DataFrame(pd.read_csv(params['testdata_path']))

    X1 = data1.drop(['Label'], axis=1)
    X7 = data7.drop(['Label'], axis=1)

    Y1 = data1.loc[:, ['Label']]
    Y7 = data7.loc[:, ['Label']]


    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    X1_scaler =  X_scaler.fit_transform(X1)
    X7_scaler =  X_scaler.fit_transform(X7)

    

    #划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X1_scaler, Y1, test_size=0.2, random_state=123)

    lgb_train = lgb.Dataset(X_train, y_train['Label'].values)
    lgb_eval = lgb.Dataset(X_test, y_test['Label'].values, reference=lgb_train)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        # 'metric': {'l2', 'auc'},  # 评估函数
        'metric': 'rmse',  # 评估函数
        'num_leaves': 126,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=10)  # ,feval=score)

    yp = gbm.predict(X7_scaler, num_iteration=gbm.best_iteration)
    yt = Y7['Label'].values

    res = {}
    res['rmse'] = np.sqrt(mean_squared_error(yt/60,yp/60))

    print(json.dumps(res))
except Exception as e:
    traceback.print_exc()





