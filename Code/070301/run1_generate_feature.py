# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:58:58 2018

@author: ASUS
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

data_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/preprocessed_data/')
train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/test.csv')
#-------agg特征-------
train_agg = pd.read_csv(data_path + 'train_agg.csv')
test_agg = pd.read_csv(data_path + 'test_agg.csv')
agg = pd.concat([train_agg,test_agg],copy=False)

train_flg = pd.read_csv(data_path + 'train_flg.csv')
test_flg = pd.DataFrame(test_agg['USRID'])
test_flg['FLAG'] = -1
flg = pd.concat([train_flg,test_flg],copy=False)

data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)

#-------log特征-------
train_log = pd.read_csv(data_path + 'train_log.csv')
test_log = pd.read_csv(data_path + 'test_log.csv')
log = pd.concat([train_log,test_log],copy=False)
#点击模块特征
#log = log.head(20)
log[['EVT_LBL_1','EVT_LBL_2','EVT_LBL_3']] = log['EVT_LBL'].str.split('-',expand=True)
#时间特征
log['time'] = log['OCC_TIM'].apply(lambda x:time.strptime(x, "%Y-%m-%d %H:%M:%S"))
log['time_sec'] = log['time'].apply(lambda x:time.mktime(x))
log['day'] = log['time'].apply(lambda x:x.tm_mday)
log['hour'] = log['time'].apply(lambda x:x.tm_hour)
log['weekday'] = log['time'].apply(lambda x:x.tm_wday)
log = log.sort_values(['USRID','time_sec'])
EVT_1_count_1 = log.loc[log['day']>=25].groupby(['USRID'],as_index=False)['EVT_LBL_1'].agg({'EVT_1_count_1':'count'})
EVT_2_count_1 = log.loc[log['day']>=25].groupby(['USRID'],as_index=False)['EVT_LBL_2'].agg({'EVT_2_count_1':'count'})
EVT_3_count_1 = log.loc[log['day']>=25].groupby(['USRID'],as_index=False)['EVT_LBL_3'].agg({'EVT_3_count_1':'count'})
EVT_1_count_2 = log.loc[log['day']>=18].groupby(['USRID'],as_index=False)['EVT_LBL_1'].agg({'EVT_1_count_2':'count'})
EVT_2_count_2 = log.loc[log['day']>=18].groupby(['USRID'],as_index=False)['EVT_LBL_2'].agg({'EVT_2_count_2':'count'})
EVT_3_count_2 = log.loc[log['day']>=18].groupby(['USRID'],as_index=False)['EVT_LBL_3'].agg({'EVT_3_count_2':'count'})
log['next_time'] = log.groupby(['USRID'])['time_sec'].diff(-1).apply(np.abs)
log_feature = log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})
data = pd.merge(data,log_feature,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_1_count_1,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_2_count_1,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_3_count_1,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_1_count_2,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_2_count_2,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_3_count_2,on=['USRID'],how='left',copy=False)

train_data = data[data['FLAG']!=-1]
test_data = data[data['FLAG']==-1]
train_data.to_csv(train_path,index=False)
test_data.to_csv(test_path,index=False)
