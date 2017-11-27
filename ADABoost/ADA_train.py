# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:38:43 2017

@author: jules
"""

import numpy as np
import pandas as pd
import pylab as plt
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from testing import testing

def ADA_train(X_train, y_train, T):
    abclf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=T, learning_rate=0.03)
    abclf.fit(X_train, y_train)
    
    return  abclf
    