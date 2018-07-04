# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:57:00 2018

@author: ASUS
"""

import os
import sys
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import lightgbm as lgb
from save_log import Logger
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values

xx_cv = []
xx_pre = []

N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)
for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=50)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,y_pred))
    xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))

s = 0
for i in xx_pre:
    s = s + i

s = s /N

res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(s)

print('xx_cv',np.mean(xx_cv))

import time
time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')