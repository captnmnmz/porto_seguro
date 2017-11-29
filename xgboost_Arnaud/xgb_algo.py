#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:13:09 2017

@author: arnaud
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from gini_comp import gini_xgb
from feature_sel import feature_selection
import xgboost as xgb


# Data loading
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#%%
target = train['target'].values
test_ids= test['id'].values
train.drop(['id','target'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)

#Feature selection
train, test = feature_selection(train, test)

### XGB modeling
params = {'eta': 0.025, 'max_depth': 4, 
          'subsample': 0.9, 'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
            'min_child_weight':100,
            'alpha':4,
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = train_test_split(train, target, test_size=0.25, random_state=99)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, 
                  verbose_eval=100, early_stopping_rounds=70)

#%%
### Submission
submission = pd.DataFrame()
submission['id'] = test_ids
submission['target'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
submission.to_csv('../output/resultss.csv',index=False)