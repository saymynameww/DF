# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:35:24 2018

@author: Administrator
"""

import os
import sys
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics
from datetime import datetime
import lightgbm as lgb
from save_log import Logger
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

train_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/preprocessed_data/train_agg.csv')
train_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/preprocessed_data/train_flg.csv')
test_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/preprocessed_data/test_agg.csv')

def F1_score(params):
    max_boost_rounds = 1000
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=max_boost_rounds,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                    )
    prediction = gbm.predict(X_test)
#    threshold = 0.4
#    prediction[prediction>=threshold]=1
#    prediction[prediction<threshold]=0
    F1 = sklearn.metrics.roc_auc_score(Y_test, prediction)
    return F1

def train_tune(train_mode):
    if train_mode == 1:
        print('Tuning params...')
        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  }
        max_F1 = F1_score(params)
        print('best F1 updated:',max_F1)
        best_params = {}
        
#        print("调参1：学习率")
#        for learning_rate in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]: 
#            print('============================',learning_rate)
#            params['learning_rate'] = learning_rate
#            F1 = F1_score(params)
#            if F1 > max_F1:
#                max_F1 = F1
#                print('best F1 updated:',max_F1)
#                best_params['learning_rate'] = learning_rate
#        if 'learning_rate' in best_params:
#            params['learning_rate'] = best_params['learning_rate']
#        else:
#            del params['learning_rate']
        params['learning_rate'] = 0.08
        
#        print("调参2：提高准确率")
#        for num_leaves in range(15,45,1): #(20,200,10)
#            for max_depth in range(3,10,1): #(3,9,1)
#                print('============================',num_leaves,max_depth)
#                params['num_leaves'] = num_leaves
#                params['max_depth'] = max_depth
#                F1 = F1_score(params)
#                if F1 > max_F1:
#                    max_F1 = F1
#                    print('best F1 updated:',max_F1)
#                    best_params['num_leaves'] = num_leaves
#                    best_params['max_depth'] = max_depth
#        if 'num_leaves' in best_params:
#            params['num_leaves'] = best_params['num_leaves']
#            params['max_depth'] = best_params['max_depth']
#        else:
#            del params['num_leaves'],params['max_depth']
        params['num_leaves'] = 23
        params['max_depth'] = 9
        
        print("调参3：降低过拟合")
        for min_data_in_leaf in range(10,800,10): #(10,200,5)
            print('============================',min_data_in_leaf)
            params['min_data_in_leaf'] = min_data_in_leaf
            F1 = F1_score(params)
            if F1 > max_F1:
                max_F1 = F1
                print('best F1 updated:',max_F1)
                best_params['min_data_in_leaf'] = min_data_in_leaf
        if 'min_data_in_leaf' in best_params:
            params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        else:
            del params['min_data_in_leaf']
        params['min_data_in_leaf'] = 220
        
#        print("调参4：采样")
#        for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#            for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#                for bagging_freq in range(0,50,5): #
#                    print('============================',feature_fraction,bagging_fraction,bagging_freq)
#                    params['feature_fraction'] = feature_fraction
#                    params['bagging_fraction'] = bagging_fraction
#                    params['bagging_freq'] = bagging_freq
#                    F1 = F1_score(params)
#                    if F1 > max_F1:
#                        max_F1 = F1
#                        print('best F1 updated:',max_F1)
#                        best_params['feature_fraction'] = feature_fraction
#                        best_params['bagging_fraction'] = bagging_fraction
#                        best_params['bagging_freq'] = bagging_freq
#        if 'feature_fraction' in best_params:
#            params['feature_fraction'] = best_params['feature_fraction']
#            params['bagging_fraction'] = best_params['bagging_fraction']
#            params['bagging_freq'] = best_params['bagging_freq']
#        else:
#            del params['feature_fraction'],params['bagging_fraction'],params['bagging_freq']
        params['feature_fraction'] = 0.9
        params['bagging_fraction'] = 0.8
        params['bagging_freq'] = 5
        
        
#        print("调参5：正则化")
#        for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#            for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
#                for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#                    print('============================',lambda_l1,lambda_l2,min_split_gain)
#                    params['lambda_l1'] = lambda_l1
#                    params['lambda_l2'] = lambda_l2
#                    params['min_split_gain'] = min_split_gain
#                    F1 = F1_score(params)
#                    if F1 > max_F1:
#                        max_F1 = F1
#                        print('best F1 updated:',max_F1)
#                        best_params['lambda_l1'] = lambda_l1
#                        best_params['lambda_l2'] = lambda_l2
#                        best_params['min_split_gain'] = min_split_gain
#        if 'lambda_l1' in best_params:
#            params['lambda_l1'] = best_params['lambda_l1']
#            params['lambda_l2'] = best_params['lambda_l2']
#            params['min_split_gain'] = best_params['min_split_gain']
#        else:
#            del params['lambda_l1'],params['lambda_l2'],params['min_split_gain']
        params['lambda_l1'] = 1
        params['lambda_l2'] = 1
        params['min_split_gain'] = 1
        
        print('Tuning params DONE, best_params:',best_params)
        return params
    elif train_mode == 2:
        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': 1, 'learning_rate': 0.07, 'num_leaves': 32, 'max_depth': 5, 'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 1, 'lambda_l2': 1}#, 'min_split_gain': 1
#        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': 1, 'learning_rate': 0.04, 'num_leaves': 18, 'max_depth': 8, 'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 1, 'lambda_l2': 1}#, 'min_split_gain': 1
        return params
    else:
        print('mode error!')


def train_func(params):
    print('Training...')
    print('All params:',params)
    cv_results = lgb.cv(train_set=lgb_train,
                         params=params,
                         nfold=5,
                         num_boost_round=1000,
                         early_stopping_rounds=5,
                         verbose_eval=False,
                         metrics=['auc'])
    optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
    print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
    print('Best cv auc = {}'.format(np.max(cv_results['auc-mean'])))
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=optimum_boost_rounds,
                    verbose_eval=False,
                    valid_sets=lgb_eval
                    )
    gbm.save_model(os.path.join(os.pardir,os.pardir,'model/lgb_model.txt'))
    return gbm

def test_threshold(gbm,threshold,f):
    validation = gbm.predict(X_test)
    validation[validation>=threshold]=1
    validation[validation<threshold]=0
    print('二分阈值：'+str(threshold)+' 验证集F1_score：' + str(sklearn.metrics.f1_score(Y_test, validation)),file=f)
    prediction = gbm.predict(test_feature)
    prediction[prediction >= threshold]=1
    prediction[prediction < threshold]=0
    prediction = list(map(int,prediction))
    print('测试集为1的个数：' + str(len(np.where(np.array(prediction)==1)[0])),file=f)
#    print('测试集为0的个数：' + str(len(np.where(np.array(prediction)==0)[0])),file=f)
    
def predict_func(gbm,show_importance):
#    print('特征阈值：'+str(importance_threshold)+' 特征数：'+ str(X_test.columns.size))
    if show_importance==1:
        print('所用特征重要性：'+ str(list(gbm.feature_importance())))
    fig, ax = plt.subplots(1, 1, figsize=[16, 40])
    lgb.plot_importance(gbm, ax=ax, max_num_features=400)
    plt.savefig('feature_importance.png')
    
    ########################## 保存结果 ############################
    prediction = gbm.predict(test_feature)
    df_result = pd.DataFrame()
    df_result['USRID'] = test_id
    df_result['RST'] = prediction
    df_result.to_csv(os.path.join(os.pardir,os.pardir,'result/lgb_result.csv'), index=False,sep='\t')
#    res.to_csv('../submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')
    
    
feature_mode = 4
R_threshold = 0.05
train_mode = 2
show_importance = 0
stdout_backup = sys.stdout
sys.stdout = Logger("train_info.txt")
print('\n')
#train_feature,train_label,test_feature,importance_threshold = feature_selection(feature_mode,R_threshold)
train_feature = pd.read_csv(train_feat_dir).drop('USRID',axis=1)
train_label = pd.read_csv(train_label_dir).FLAG
test_feature = pd.read_csv(test_feat_dir).drop('USRID',axis=1)
test_id = pd.read_csv(test_feat_dir).USRID
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
params = train_tune(train_mode)
gbm = train_func(params) 
predict_func(gbm,show_importance) 
sys.stdout = stdout_backup
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
