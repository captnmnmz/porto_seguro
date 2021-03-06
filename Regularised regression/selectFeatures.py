# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:26:22 2017

@author: jules
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def selectFeatures(_X_train,_y_train, _X_test):
    X_train=_X_train.copy()
    y_train=_y_train.copy()
    X_test=_X_test.copy()
    #STANDARDISATION + NORMALISATION
    scaler=StandardScaler()
    X_train_ar=scaler.fit_transform(X_train)
    X_test_ar=scaler.transform(X_test)
    X_train=pd.DataFrame(X_train_ar)
    
    feat_labels = X_train.columns
    
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    ls = Lasso()
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    
    indices = np.argsort(rf.feature_importances_)[::-1]
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
    
    sfm = SelectFromModel(rf, threshold='median', prefit=True)
    print('Number of features before selection: {}'.format(X_train.shape[1]))
    n_features = sfm.transform(X_train).shape[1]
    print('Number of features after selection: {}'.format(n_features))
    selected_vars = list(feat_labels[sfm.get_support()])
    
    _train = _train[selected_vars + ['id','target']]
    
    return X_train,y_train,X_test selected_vars