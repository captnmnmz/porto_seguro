# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:13:05 2017

@author: jules
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split



from ensemble import Ensemble
from createMeta import createMeta
from createPoly import createPoly
from drop import drop
from dummy import dummy
from selectFeatures import selectFeatures
from target_encode import target_encode
from testing import testing
from undersampling import undersampling


#%%
## Load the dataset
data_train_pd=pd.read_csv("train.csv")
data_test_pd=pd.read_csv("test.csv")

#%%
#non linear interaction variables
fe = 1
print(fe)
if (fe==1):
    data_train_pd['v003'] = data_train_pd["ps_ind_03"]+data_train_pd["ps_ind_14"]+np.square(data_train_pd["ps_ind_15"])
    data_train_pd['v009'] = data_train_pd["ps_ind_03"]+data_train_pd["ps_ind_14"]+np.tanh(data_train_pd["ps_ind_15"])
    data_train_pd['v015'] = data_train_pd["ps_reg_01"]+data_train_pd["ps_reg_02"]**3+data_train_pd["ps_reg_03"]
    data_train_pd['v020'] = data_train_pd["ps_reg_01"]**2+np.tanh(data_train_pd["ps_reg_02"])+data_train_pd["ps_reg_03"]**2.5
    data_train_pd['v079'] = data_train_pd["ps_calc_01"]+data_train_pd["ps_calc_13"]+np.tanh(data_train_pd["ps_calc_14"])
    
    data_test_pd['v003'] = data_test_pd["ps_ind_03"]+data_test_pd["ps_ind_14"]+np.square(data_test_pd["ps_ind_15"])
    data_test_pd['v009'] = data_test_pd["ps_ind_03"]+data_test_pd["ps_ind_14"]+np.tanh(data_test_pd["ps_ind_15"])
    data_test_pd['v015'] = data_test_pd["ps_reg_01"]+data_test_pd["ps_reg_02"]**3+data_test_pd["ps_reg_03"]
    data_test_pd['v020'] = data_test_pd["ps_reg_01"]**2+np.tanh(data_test_pd["ps_reg_02"])+data_test_pd["ps_reg_03"]**2.5
    data_test_pd['v079'] = data_test_pd["ps_calc_01"]+data_test_pd["ps_calc_13"]+np.tanh(data_test_pd["ps_calc_14"])
#%%
###PREPROCESSING
#storing useful metadata
meta = createMeta(data_train_pd)
print('for train data : ')
print(meta.info)
#%%
#PREPROCESSING TEST DATA
###PREPROCESSING
#storing useful metadata
meta_test = createMeta(data_test_pd)
print('for test data :')
print(meta_test.info)
#%%
###UNDERSAMPLING of the records with target=0
data_train_pd=undersampling(data_train_pd)

#%%
###DROPPING train data with too many missing values, replacing with mean for the other missing values
print('Before dropping train contains {} variables '.format(data_train_pd.shape[1]))
data_train_pd=drop(data_train_pd,meta)
print('After dropping train contains {} variables '.format(data_train_pd.shape[1]))
###DROPPING the same parts in test data, replacing with mean for the other missing values
print('Before dropping test contains {} variables '.format(data_test_pd.shape[1]))
data_test_pd=drop(data_test_pd,meta_test)
print('After dropping train contains {} variables '.format(data_test_pd.shape[1]))

#%%
#smooth the values in pas_car_11_cat
train_encoded, test_encoded = target_encode(data_train_pd["ps_car_11_cat"], 
                             data_test_pd["ps_car_11_cat"], 
                             target=data_train_pd.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
data_train_pd['ps_car_11_cat_te'] = train_encoded
data_train_pd.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
data_test_pd['ps_car_11_cat_te'] = test_encoded
data_test_pd.drop('ps_car_11_cat', axis=1, inplace=True)
meta_test.loc['ps_car_11_cat','keep'] = False  # Updating the meta    
#%%
#We create dummy variables to deal with the categorical variables
print('Before dummification train contains {} variables '.format(data_train_pd.shape[1]))
data_train_pd=dummy(data_train_pd,meta)
print('After dummification train contains {} variables '.format(data_train_pd.shape[1]))
print('Before dummification test contains {} variables '.format(data_test_pd.shape[1]))
data_test_pd=dummy(data_test_pd,meta_test)
print('After dummification test contains {} variables '.format(data_test_pd.shape[1]))
#%%
#We create polynomial interaction variables for the continous variables
"""print('Before creating interactions we have {} variables in train'.format(data_train_pd.shape[1]))
data_train_pd=createPoly(data_train_pd, meta)
print('After creating interactions we have {} variables in train'.format(data_train_pd.shape[1]))
print('Before creating interactions we have {} variables in test'.format(data_test_pd.shape[1]))
data_test_pd=createPoly(data_test_pd, meta_test)
print('After creating interactions we have {} variables in test'.format(data_test_pd.shape[1]))"""

#%%
#FEATURE SELECTION : RANDOM FOREST
#data_train_pd, selected_vars=selectFeatures(data_train_pd)
#data_test_pd = data_test_pd[selected_vars + ['id']]


#%%
# The first two columns contains the exam scores and the third column contains the label.
# TODO NORMALISATION
_target_train = data_train_pd.target.values
_train = data_train_pd.drop(['id','target'], axis=1)

train, valid, target_train, target_valid = train_test_split(_train, _target_train, test_size=0.3, random_state=None)



id_test = data_test_pd.id.values
test = data_test_pd.drop(['id'], axis=1)





#%%

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10  
lgb_params['min_child_samples'] = 500
lgb_params['feature_fraction'] = 0.9
lgb_params['bagging_freq'] = 1
lgb_params['seed'] = 200

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['feature_fraction'] = 0.8
lgb_params2['bagging_freq'] = 1
lgb_params2['seed'] = 200


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['feature_fraction'] = 0.95
lgb_params3['bagging_freq'] = 1
lgb_params3['seed'] = 200

# XGB params
_xgb_params = {'eta': 0.025, 'max_depth': 4, 
          'subsample': 0.9, 'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
            'min_child_weight':100,
            'alpha':4,
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
           
lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

#%%


log_model = LogisticRegression()

#%%     
stack = Ensemble(n_splits=6,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3),
                 xgb_params=_xgb_params)
        
y_pred, y_valid = stack.fit_predict(train, target_train, valid, target_valid, test) 

#%%
#GINI TEST
gini=testing(target_valid, y_valid)
print("gini score is :")
print(gini)       

#%%
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('submit.csv', index=False)
     