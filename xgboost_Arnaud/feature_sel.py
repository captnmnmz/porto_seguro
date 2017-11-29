#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:13:09 2017

@author: arnaud
"""
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

data_median = 0
data_mean = 0
feature_values = 0

#Reconstruction of ps_reg_03 in Federations and Municipalities
def recon(reg):
    I = int(np.round((40*reg)**2)) #ps_reg_03 = sqrt(I)/40
    for f in range(28): #there are 27 federations
        if (I - f) % 27 == 0: #I = 27M + F
            F = f
    M = (I - F)//27
    return F, M

#Creation of new features
def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    #Creation of a feature combining the two most relevant features
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    #Creation of a feature with the number of -1
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    #Creation of binary features with the position regarding mean and median
    for c in dcol:
        if '_bin' not in c:
            df[c+str('_median_range')] = (df[c].values > data_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > data_mean[c]).astype(np.int)
    #Creation of binary feature for each value possible
    for feature in feature_values:
        #non binary nor continuous feature, up to 13 values
        if len(feature_values[feature])>2 and len(feature_values[feature]) < 14:
            for value in feature_values[feature]:
                df[c+'_oh_' + str(value)] = (df[c].values == value).astype(np.int)
    return df

#Use of a pool of threads to transform the datasets more efficiently
def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

#Feature engineering
def feature_selection(train, test):
    unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(unwanted, axis=1)  
    test = test.drop(unwanted, axis=1)
    #Creation of two new features by reconstruction of ps_reg_03
    train['ps_reg_F'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
    train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
    test['ps_reg_F'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
    test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
    
    #Computing of mean and median without the -1
    train = train.replace(-1, np.NaN)
    global data_median
    data_median = train.median(axis=0)
    global data_mean
    data_mean = train.mean(axis=0)
    train = train.fillna(-1)
    global feature_values
    feature_values = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}
    train = multi_transform(train)
    test = multi_transform(test)
    return train, test
    

