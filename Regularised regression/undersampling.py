# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:37:59 2017

@author: jules
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

def undersampling(train):
    _train=train.copy()
    desired_apriori=0.10
    
    # Get the indices per target value
    idx_0 = _train[_train.target == 0].index
    idx_1 = _train[_train.target == 1].index
    
    # Get original number of records per target value
    nb_0 = len(_train.loc[idx_0])
    nb_1 = len(_train.loc[idx_1])
    
    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
    undersampled_nb_0 = int(undersampling_rate*nb_0)
    print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
    print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))
    
    # Randomly select records with target=0 to get at the desired a priori
    undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)
    
    # Construct list with remaining indices
    idx_list = list(undersampled_idx) + list(idx_1)
    
    # Return undersample data frame
    _train = _train.loc[idx_list].reset_index(drop=True)
    
    return _train