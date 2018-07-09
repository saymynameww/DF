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
#train_agg = pd.read_csv(data_path + 'train_agg.csv',nrows=500)
#test_agg = pd.read_csv(data_path + 'test_agg.csv',nrows=500)
agg = pd.concat([train_agg,test_agg],copy=False)

train_flg = pd.read_csv(data_path + 'train_flg.csv')
#train_flg = pd.read_csv(data_path + 'train_flg.csv',nrows=500)
test_flg = pd.DataFrame(test_agg['USRID'])
test_flg['FLAG'] = -1
flg = pd.concat([train_flg,test_flg],copy=False)

data = pd.merge(flg,agg,on=['USRID'],how='left',copy=False)

#-------log特征-------
train_log = pd.read_csv(data_path + 'train_log.csv')
test_log = pd.read_csv(data_path + 'test_log.csv')
#train_log = pd.read_csv(data_path + 'train_log.csv',nrows=500)
#test_log = pd.read_csv(data_path + 'test_log.csv',nrows=500)
log = pd.concat([train_log,test_log],copy=False)
#点击模块特征
#log = log.head(20)
end_time = '2018-03-31 23:59:59'
log[['EVT_1','EVT_2','EVT_3']] = log['EVT_LBL'].str.split('-',expand=True)
log['time'] = log['OCC_TIM'].apply(lambda x:time.strptime(x, "%Y-%m-%d %H:%M:%S"))
log['total_second'] = log['time'].apply(lambda x:time.mktime(x))
log['total_minute'] = log['total_second']/60
log['total_hour'] = log['total_minute']/60
log['day'] = log['time'].apply(lambda x:x.tm_mday)
log['hour'] = log['time'].apply(lambda x:x.tm_hour)
log['weekday'] = log['time'].apply(lambda x:x.tm_wday) + 1
log = log.sort_values(['USRID','total_second'])

#时间特征
day_info = log.groupby(['USRID'],as_index=False)['day'].agg({'day_mean':np.mean,'day_std':np.std,'day_min':np.min,'day_max':np.max,'day_skew':lambda x: pd.Series.skew(x),'day_kurt':lambda x: pd.Series.kurt(x)}) 
day_info['day_rest'] = (day_info['day_max']-31).apply(np.abs)
hour_info = log.groupby(['USRID'],as_index=False)['total_hour'].agg({'hour_mean':np.mean,'hour_std':np.std,'hour_min':np.min,'hour_max':np.max,'hour_skew':lambda x: pd.Series.skew(x),'hour_kurt':lambda x: pd.Series.kurt(x)}) 
hour_info['hour_rest'] = (hour_info['hour_max']-np.max(log['total_second'])).apply(np.abs)
#行为相隔时间特征
log['next_second'] = log.groupby(['USRID'])['total_second'].diff(-1).apply(np.abs)
next_time = log.groupby(['USRID'],as_index=False)['next_second'].agg({'next_second_mean':np.mean,'next_second_std':np.std,'next_second_min':np.min,'next_second_max':np.max,'next_second_skew':lambda x: pd.Series.skew(x),'next_second_kurt':lambda x: pd.Series.kurt(x)}) 
next_time['next_minute_mean'] = next_time['next_second_mean']/60
next_time['next_minute_std'] = next_time['next_second_std']/60
next_time['next_minute_min'] = next_time['next_second_min']/60
next_time['next_minute_max'] = next_time['next_second_max']/60
next_time['next_minute_skew'] = next_time['next_second_skew']/60
next_time['next_minute_kurt'] = next_time['next_second_kurt']/60
next_time['next_hour_mean'] = next_time['next_minute_mean']/60
next_time['next_hour_std'] = next_time['next_minute_std']/60
next_time['next_hour_min'] = next_time['next_minute_min']/60
next_time['next_hour_max'] = next_time['next_minute_max']/60
next_time['next_hour_kurt'] = next_time['next_minute_kurt']/60
next_time['next_hour_skew'] = next_time['next_minute_mean']/60
next_time['next_day_mean'] = next_time['next_hour_mean']/24
next_time['next_day_std'] = next_time['next_hour_std']/24
next_time['next_day_min'] = next_time['next_hour_min']/24
next_time['next_day_max'] = next_time['next_hour_max']/24
next_time['next_day_skew'] = next_time['next_hour_skew']/24
next_time['next_day_kurt'] = next_time['next_hour_kurt']/24
#滑窗 窗口内行为次数
EVT_count_31to31 = log.loc[log['day']>=31].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_31to31':'count'})
EVT_count_30to31 = log.loc[log['day']>=30].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_30to31':'count'})
EVT_count_29to31 = log.loc[log['day']>=29].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_29to31':'count'})
EVT_count_25to31 = log.loc[log['day']>=25].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_25to31':'count'})
EVT_count_18to31 = log.loc[log['day']>=18].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_18to31':'count'})
EVT_count_11to31 = log.loc[log['day']>=11].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_11to31':'count'})
EVT_count_4to31 = log.loc[log['day']>=4].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_4to31':'count'})
EVT_count_1to31 = log.loc[log['day']>=1].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_1to31':'count'})
EVT_count_1to3 = log.loc[(log['day']>=1)&(log['day']<=3)].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_1to3':'count'})
EVT_count_1to7 = log.loc[(log['day']>=1)&(log['day']<=7)].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_1to7':'count'})
#MERGE

data = pd.merge(data,day_info,on=['USRID'],how='left',copy=False)
data = pd.merge(data,hour_info,on=['USRID'],how='left',copy=False)
data = pd.merge(data,next_time,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_31to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_30to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_29to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_25to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_18to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_11to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_4to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_1to31,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_1to3,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_1to7,on=['USRID'],how='left',copy=False)
data = data.fillna(0)

train_data = data[data['FLAG']!=-1]
test_data = data[data['FLAG']==-1]
train_data.to_csv(train_path,index=False)
test_data.to_csv(test_path,index=False)
