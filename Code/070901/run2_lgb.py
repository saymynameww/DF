# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:43:35 2018

@author: ASUS
"""

import os
import sys
from datetime import datetime
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from save_log import Logger

def train_cv(params):
    N = 5
    model_i = 1
    print('All params:',params)
    skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=624)
    for train_in,test_in in skf.split(train_feature,train_label):
        X_train,X_test,y_train,y_test = train_feature[train_in],train_feature[test_in],train_label[train_in],train_label[test_in]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        early_stopping_rounds=50)
    
        gbm.save_model(model_path+'model_'+str(model_i)+'.txt')
        model_i += 1
    
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        cv_roc.append(roc_auc_score(y_test,y_pred))
        cv_prediction.append(gbm.predict(test, num_iteration=gbm.best_iteration))

def predict_func():
    best_cv = 0
    best_cv_roc = cv_roc[0]
    best_cv_prediction = cv_prediction[0]
    for i in range(1,len(cv_roc)):
        if cv_roc[i] > best_cv_roc:
            best_cv = i
            best_cv_roc = cv_roc[i]
            best_cv_prediction = cv_prediction[i]
    print('best_cv:'+str(best_cv)+' best_cv_roc:',best_cv_roc)
    result = pd.DataFrame()
    result['USRID'] = list(test_userid.values)
    result['RST'] = list(best_cv_prediction)
    time_date = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    submit_file_name = '%s_%s.csv'%(str(time_date),str(best_cv_roc).split('.')[1])
    result.to_csv(submit_file_name,index=False,sep='\t')
    
    gbm = lgb.Booster(model_file=model_path+'model_'+str(best_cv)+'.txt')
#    print('特征阈值：'+str(importance_threshold)+' 特征数：'+ str(X_test.columns.size))
    if show_importance==1:
        print('所用特征重要性：'+ str(list(gbm.feature_importance())))
    fig, ax = plt.subplots(1, 1, figsize=[16, 40])
    lgb.plot_importance(gbm, ax=ax, max_num_features=400)
    plt.savefig('feature_importance.png')
    print(submit_file_name+' 线上:{}')
    
if __name__ == "__main__":
    time_start = datetime.now()
    print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))
    print('Loading data...')
    model_path = os.path.join(os.pardir,os.pardir, 'Model/')
    train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/train.csv')
    test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train_user_id = train.pop('USRID')
    train_label = train.pop('FLAG')
    col = train.columns
    train_feature = train[col].values
    test_userid = test.pop('USRID')
    test.pop('FLAG')
    test_feature = test[col].values
    
    #feature_mode = 4
    #R_threshold = 0.05
    #train_mode = 2
    show_importance = 0
    stdout_backup = sys.stdout
    sys.stdout = Logger("train_info.txt")
    print('\n')
    #train_feature,train_label,test_feature,importance_threshold = feature_selection(feature_mode,R_threshold)
    #params = train_tune(train_mode)
#    params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': {'auc'}, 'num_leaves': 32, 'learning_rate': 0.01, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': 0}
    params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': {'auc'}, 'num_leaves': 32, 'learning_rate': 0.01, 'verbose': 0}
    cv_roc = []
    cv_prediction = []
    train_cv(params)
    predict_func() 
    sys.stdout = stdout_backup
    time_end = datetime.now()
    print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')