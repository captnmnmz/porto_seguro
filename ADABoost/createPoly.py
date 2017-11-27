# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:13:59 2017

@author: jules
"""
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
def createPoly(set,meta):
    v = meta[(meta.level == 'interval') & (meta.keep)].index
    poly = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
    interactions = pd.DataFrame(data=poly.fit_transform(set[v]), columns=poly.get_feature_names(v))
    interactions.drop(v, axis=1, inplace=True)  
    
    
    _set = pd.concat([set, interactions], axis=1)
    return _set
    
