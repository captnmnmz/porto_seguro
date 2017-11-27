# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:50:30 2017

@author: jules
"""
from sklearn.preprocessing import Imputer
def drop(train, meta):
    _train=train.copy()
    # Dropping the variables with too many missing values
    vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
    _train.drop(vars_to_drop, inplace=True, axis=1)
    meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta
    
    # Imputing with the mean or mode for other major missin features
    mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
    mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
    _train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
    _train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
    _train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
    _train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()

    return _train