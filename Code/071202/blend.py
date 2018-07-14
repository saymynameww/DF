# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:05:32 2018

@author: Administrator
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/test.csv')
train = pd.read_csv(train_path).fillna(0)
test = pd.read_csv(test_path).fillna(0)
train.pop('USRID')
train_label = train.pop('FLAG')
col = train.columns
test_userid = test.pop('USRID')
test.pop('FLAG')

result1 = np.load('total_rf_result.npy')
result2 = np.load('total_lr_result.npy')
result3 = np.load('total_lgb_result.npy')
#label = np.load('test_label.npy')
result_blend = 0.025*result1+0.015*result2+0.96*result3

result = pd.DataFrame()
result['USRID'] = list(test_userid.values)
result['RST'] = list(result_blend)
time_date = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
submit_file_name = 'blend_result_0713.csv'
result.to_csv(submit_file_name,index=False,sep='\t')

#result_roc=roc_auc_score(label,result_blend)