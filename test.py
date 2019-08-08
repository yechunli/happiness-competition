import numpy as np
import pandas as pd

happiness_data = 'F:\\happiness_data\\happiness_train_abbr.csv'

train_data = pd.read_csv(happiness_data)


train_data=train_data[~train_data['happiness'].isin([-8])]
train_data.reset_index(inplace=True)
train_data.drop('index', axis=1, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
for train_index, test_index in split.split(train_data, train_data['happiness']):
    train_set = train_data.loc[train_index]
    test_set = train_data.loc[test_index]
y_train = train_set['happiness']
x_train = train_set.drop('happiness', axis=1)
y_test = test_set['happiness']
x_test = test_set.drop('happiness', axis=1)
#x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.25)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

x_enc = OneHotEncoder()
y_enc = OneHotEncoder()
mixScaler = MinMaxScaler()
stdScaler = StandardScaler()

def data_processing(x, y, mode):
    x.drop(['id', 'survey_time', 'work_yr', 'work_type', 'work_manage', 'work_status'], axis=1, inplace=True)
    x = x.replace('', np.nan)
    x[x<0] = np.nan
    median = x.median()
    x = x.fillna(median)
    x['person_income'] = x['family_income'] / x['family_m']
    x_onehot = x.loc[:, ['status_peer', 'status_3_before', 'car', 'work_exper', 'survey_type', 'city', 'religion', 'province', 'county', 'gender', 'nationality', 'political', 'hukou', 'marital']]
    x_original = x.drop(['status_peer', 'status_3_before', 'car', 'work_exper', 'survey_type', 'city', 'religion', 'province', 'county', 'gender', 'nationality', 'political', 'hukou', 'marital'],axis=1)
    if mode == 'train':
        x_enc.fit(x_onehot)
        mixScaler.fit(x_original)
        y_enc.fit(np.reshape(y, newshape=[-1,1]))
    x1 = x_enc.transform(x_onehot).toarray()
    x2 = mixScaler.transform(x_original)
    x_input = np.c_[x1, x2]
    #y_input = y_enc.transform(np.reshape(y, newshape=[-1,1])).toarray()
    return x_input

x_input = data_processing(x_train, y_train, mode='train')
x = data_processing(x_test, y_test, mode='test')

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(x_input, y_train)
pred = tree.predict(x)
mmse = np.mean((y_test - pred)**2)

%matplotlib inline
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
# 算法参数
params = {
    'booster': 'gbtree',
#    'objective': 'multi:softmax',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
#    'num_class': 5,
    'gamma': 0.2,
    'max_depth': 5,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'scale_pos_weight' : 1,
    'eta': 0.005,
    #'seed': 1000,
    'nthread': 4,
}

plst = params.items()


dtrain = xgb.DMatrix(x_input, y_train-1) # 生成数据集格式
watchlist = [(dtrain, 'train')]
num_rounds = 20000
model = xgb.train(plst, dtrain, num_rounds, evals=watchlist, verbose_eval=100) # xgboost模型训练

# 对测试集进行预测
dtest = xgb.DMatrix(x)
y_ = model.predict(dtest)
mmse = np.mean((y_test - y_ - 1) ** 2)
mmse
#plot_importance(model)
#plt.show()

import xgboost as xgb
clf = xgb.XGBRegressor(silent=True, learning_rate=0.1, min_child_weight=1, max_depth=6,
                    gamma=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
                    scale_pos_weight=1, objective= 'reg:linear', n_estimators=100,
                    eval_metric='rmse')

clf.fit(x_input, y_train-1)
y__ = clf.predict(x)
mmse = np.mean((y_test - y__ - 1) ** 2)

