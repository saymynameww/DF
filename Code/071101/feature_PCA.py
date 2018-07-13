# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 23:22:38 2018

@author: Administrator
"""

import os
from sklearn.decomposition import PCA

train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
pca.fit(X)
pca = PCA(n_components=2)
