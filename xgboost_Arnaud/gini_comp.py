#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:13:09 2017

@author: arnaud
"""
import pandas as pd
import numpy as np
import gc

# Gini computation
def ginic(true, pred):
    size = len(true)
    true = np.asarray(true) 
    sorted_true = true[np.argsort(pred)]
    cum_true = sorted_true.cumsum()
    giniSum = cum_true.sum() / sorted_true.sum() - (size + 1) / 2.0
    return giniSum / size


def gini_normalized(true, pred):
    if pred.ndim == 2:
        pred = pred[:,1] 
    return ginic(true, pred) / ginic(true, true)


def gini_xgb(pred, data_train):
    labels = data_train.get_label()
    return 'gini', gini_normalized(labels, pred)