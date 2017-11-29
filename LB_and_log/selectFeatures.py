# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:26:22 2017

@author: jules
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
def selectFeatures(train):
    _train=train.copy()
    X_train = _train.drop(['id', 'target'], axis=1)
    y_train = _train['target']
    
    feat_labels = X_train.columns
    
    rf = RandomForestClassifier(n_estimators=700, random_state=0, n_jobs=-1)
    
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    
    indices = np.argsort(rf.feature_importances_)[::-1]
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
    
    sfm = SelectFromModel(rf, threshold=0.002, prefit=True)
    print('Number of features before selection: {}'.format(X_train.shape[1]))
    n_features = sfm.transform(X_train).shape[1]
    print('Number of features after selection: {}'.format(n_features))
    selected_vars = list(feat_labels[sfm.get_support()])
    
    _train = _train[selected_vars + ['id','target']]
    
    return _train, selected_vars