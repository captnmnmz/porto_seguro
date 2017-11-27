#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:06:12 2017

@author: bastienchevallier
"""
# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>

#%%
import numpy as np
import pandas as pd
import pylab as plt

from read_dataset import read_dataset
from read_dataset import read_dataset_acc
from results import results
from time import sleep
from sklearn.tree import DecisionTreeClassifier

#%%
# =========================================== TRAINING AND TEST on training test ==============================

###### Values to set #######
size_training = 30000     # on 595212
size_test = 10000
############################

instances_training, targets_training, zero_test, zero_res, one_test, one_res = read_dataset_acc(size_training, size_test)
num_labels         = 2 #0 or 1

### Fetch the data and load it in pandas
#train = pd.read_csv("../../../train.csv")
#test = pd.read_csv("../../../test.csv")
#print("Size of the train set: ", train.shape)
#print("Size of the test set: ", test.shape)

# See data (five rows) using pandas tools
#print data.head(2)


### Prepare input to scikit and train and test cut

#binary_data = [np.logical_or(data['Cover_Type'] == 1,data['Cover_Type'] == 2)] # two-class classification set
#X = train.drop('target', axis=1).values
#y = train['target'].values

# Import cross validation tools from scikit
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)
X_train = instances_training
y_train = targets_training
X0_test = zero_test
y0_test = zero_res
X1_test = one_test
y1_test = one_res

#%%
# Compute accuracy of the classifier (correctly classified instances)
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, y_pred)

#%%
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

def AdaBoost(D, T):
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    training_scores = np.zeros(X_train.shape[0])      # N.B. training prediction (without sign fn)
    test0_scores = np.zeros(X0_test.shape[0])           # N.B. test prediction (without sign fn)
    test1_scores = np.zeros(X1_test.shape[0])
    
    training_errors = []
    test_errors = []

    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=D)
        clf.fit(X_train, y_train, sample_weight=w)
        y_pred = clf.predict(X_train)
        
        indicator = np.not_equal(y_pred, y_train)
        print("indicator = ", indicator)
        gamma = w[indicator].sum() / w.sum()
        print("gamma = ", gamma)
        alpha = np.log((1-gamma) / gamma)
        print("alpha = ", alpha)
        w *= np.exp(alpha*indicator) 
        print("y_pred ", y_pred)
        print("alpha  ", alpha)
        
        training_scores += alpha*y_pred
        print("training_scores*y_train ", training_scores*y_train)
        training_error = 1.*len(training_scores[training_scores*y_train<0]) / len(X_train)
        print("training_scores ", training_scores)
        print("training_error ", training_error)
        
        y0_test_pred = clf.predict(X0_test)
        y1_test_pred = clf.predict(X1_test)
        
        test0_scores += alpha * y0_test_pred
        test1_scores += alpha * y1_test_pred
        print("test0_scores ", test0_scores)
        print("test1_scores ", test1_scores)
        
        test0_error =len(test0_scores[test0_scores*y0_test<0]) / len(X0_test)
        print(" NUM = ", len(test0_scores[test0_scores*y0_test<0]))
        test1_error =len(test1_scores[test1_scores * y1_test < 0]) / len(X1_test)

        
        test_error = test0_error*(X0_test.shape[0]/(X0_test.shape[0]+X1_test.shape[0]))+test1_error*(X1_test.shape[0]/(X0_test.shape[0]+X1_test.shape[0]))
        
        training_errors.append(training_error)
        test_errors.append(test_error)

    return test_error



Ds = [5, 8, 10, 12, 15]
NumTrees = [i for i in range(100,550,50)]
final_test_errors = [[] for i in range(len(Ds))]
for i in range (len(Ds)):
    for j in range (len(NumTrees)) :
        print("===================================================")
        print("Tree Depth : ", Ds[i])
        print("Number of trees : ", NumTrees[j])
        final_test_errors[i].append(AdaBoost(Ds[i], NumTrees[j]))
        print("Final_test_error : ", final_test_errors)
        print("===================================================")


mini = min(final_test_errors[0])
index1 = 0
index2 = final_test_errors[0].index(mini)
for i in range(1,final_test_errors.length()):
    mini_ = min(final_test_errors[i])
    if(mini > mini_ ):
        index1 = i
        index2 = final_test_errors[i].index(mini_)
        mini = mini_
        
Best_D = Ds[index1]
Best_numtrees = NumTrees[index2]
print("================= OPTIMIZATION RESULTS ================")
print ("Best configuration : ")
print ("Tree depth : ", Best_D)
print ("Number of tree : ", Best_numtrees)
print ("Error : ", final_test_errors[index1][index2])
print("=======================================================")

#%%
# =========================================== TEST on test set with the best configuration ==============================
print("=============== COMPILATION ON TEST set =======================")

X_train, y_train, X_test = read_dataset(size_training)

def AdaBoost_pred(D, T):
    w = np.ones(X_train.shape[0]) / X_train.shape[0]

    for t in range(T):
        print("Tree N° : ", t)
        clf = DecisionTreeClassifier(max_depth=D)
        clf.fit(X_train, y_train, sample_weight=w)
        y_pred = clf.predict(X_train)
        
        indicator = np.not_equal(y_pred, y_train)
        gamma = w[indicator].sum() / w.sum()
        alpha = np.log((1-gamma) / gamma)
        w *= np.exp(alpha * indicator) 

        y_test_pred = clf.predict(X_test)

    return y_test_pred

pred = AdaBoost_pred(8, 100)
results(pred)






