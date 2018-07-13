# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:05:32 2018

@author: Administrator
"""

import numpy as np
from sklearn.metrics import roc_auc_score

result1 = np.load('rf_result_0.8301961465693657.npy')
result2 = np.load('lr_result_0.8354612576204143.npy')
result3 = np.load('lgb_result_0.8344954331141057.npy')
label = np.load('test_label.npy')
result_blend = 0.3*result1+0.2*result2+0.5*result3

result_roc=roc_auc_score(label,result_blend)