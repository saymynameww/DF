# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:06:36 2018

@author: Administrator
"""

#EVT_count = pd.DataFrame(agg['USRID'])
#for i in range(2):
#    log_i = log[i*1000:(i+1)*1000]
#    label_encoder = LabelEncoder()
#    integer_encoded1 = label_encoder.fit_transform(log_i.EVT_1)
#    integer_encoded2 = label_encoder.fit_transform(log_i.EVT_2)
#    integer_encoded3 = label_encoder.fit_transform(log_i.EVT_3)
#    integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
#    integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
#    integer_encoded3 = integer_encoded3.reshape(len(integer_encoded3), 1)
#    onehot_encoder = OneHotEncoder(sparse=False)
#    onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1)
#    onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)
#    onehot_encoded3 = onehot_encoder.fit_transform(integer_encoded3)
#    log_array = np.hstack((log_i.values,onehot_encoded1,onehot_encoded2,onehot_encoded3))
#    EVT_list = np.hstack((['EVT_1_'+str(x) for x in range(onehot_encoded1.shape[1])],['EVT_2_'+str(x) for x in range(onehot_encoded2.shape[1])],['EVT_3_'+str(x) for x in range(onehot_encoded3.shape[1])]))
#    log_cols = np.hstack((log_i.columns,EVT_list))
#    log_i = pd.DataFrame(log_array,columns=log_cols)
#    log_i = log_i.infer_objects()
#    for evt in EVT_list:
#        evt_count = evt+'_count'
#        EVT_count_temp = log_i[log_i[evt]==1].groupby(['USRID'],as_index=False)[evt].agg({evt_count:'count'})
#        if evt_count in EVT_count.columns:
#            EVT_count[evt_count] = EVT_count[evt_count] + EVT_count_temp[evt_count]
#        else:
#            EVT_count = pd.merge(EVT_count,EVT_count_temp,on=['USRID'],how='left',copy=False)


integer_encoded2 = label_encoder.fit_transform(log.EVT_2)
integer_encoded3 = label_encoder.fit_transform(log.EVT_3)

integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
integer_encoded3 = integer_encoded3.reshape(len(integer_encoded3), 1)

onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1)
onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)
log_array = np.hstack((log.values,onehot_encoded1,onehot_encoded2))
EVT_list = np.hstack((['EVT_1_'+str(x) for x in range(onehot_encoded1.shape[1])],['EVT_2_'+str(x) for x in range(onehot_encoded2.shape[1])]))
del onehot_encoded1,onehot_encoded2
gc.collect()
onehot_encoded3 = onehot_encoder.fit_transform(integer_encoded3)
log_array = np.hstack((log_array,onehot_encoded3))
EVT_list = np.hstack((EVT_list,['EVT_3_'+str(x) for x in range(onehot_encoded3.shape[1])]))
del onehot_encoded3
gc.collect()
#log_array = np.hstack((log.values,onehot_encoded1,onehot_encoded2,onehot_encoded3))
#EVT_list = np.hstack((['EVT_1_'+str(x) for x in range(onehot_encoded1.shape[1])],['EVT_2_'+str(x) for x in range(onehot_encoded2.shape[1])],['EVT_3_'+str(x) for x in range(onehot_encoded3.shape[1])]))
log_cols = np.hstack((log.columns,EVT_list))
log = pd.DataFrame(log_array,columns=log_cols)
log = log.infer_objects()
EVT_count = pd.DataFrame(agg['USRID'])
for evt in EVT_list:
    EVT_count_temp = log[log[evt]==1].groupby(['USRID'],as_index=False)[evt].agg({evt+'_count':'count'})
    EVT_count = pd.merge(EVT_count,EVT_count_temp,on=['USRID'],how='left',copy=False) 