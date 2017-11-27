from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from predict import *
from mapFeatures import mapFeatures
from sklearn.preprocessing import StandardScaler
from computeCostreg import computeCostreg
from computeGradreg import computeGradreg
from testing import testing
from plotBoundary import plotBoundary
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from undersampling import undersampling
from drop import drop
from dummy import dummy
from createMeta import createMeta
from createPoly import createPoly
from selectFeatures import selectFeatures
import numpy as np
import pandas as pd
#%%
## Load the dataset
data_train_pd=pd.read_csv("train.csv")
data_test_pd=pd.read_csv("test.csv")

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
#We create dummy variables to deal with the categorical variables
print('Before dummification train contains {} variables '.format(data_train_pd.shape[1]))
data_train_pd=dummy(data_train_pd,meta)
print('After dummification train contains {} variables '.format(data_train_pd.shape[1]))
print('Before dummification test contains {} variables '.format(data_test_pd.shape[1]))
data_test_pd=dummy(data_test_pd,meta_test)
print('After dummification test contains {} variables '.format(data_test_pd.shape[1]))
#%%
#We create polynomial interaction variables for the continous variables
print('Before creating interactions we have {} variables in train'.format(data_train_pd.shape[1]))
data_train_pd=createPoly(data_train_pd, meta)
print('After creating interactions we have {} variables in train'.format(data_train_pd.shape[1]))
print('Before creating interactions we have {} variables in test'.format(data_test_pd.shape[1]))
data_test_pd=createPoly(data_test_pd, meta_test)
print('After creating interactions we have {} variables in test'.format(data_test_pd.shape[1]))

#%%
#FEATURE SELECTION : RANDOM FOREST
data_train_pd, selected_vars=selectFeatures(data_train_pd)
data_test_pd = data_test_pd[selected_vars + ['id']]
#%%
# The first two columns contains the exam scores and the third column contains the label.
# NORMALISATION
y = data_train_pd.target.values
X_pd = data_train_pd.drop(['id','target'], axis=1)
X = X_pd.values

#CREATE CROSS VALIDATION ENSEMBLES
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=None)

test_id = data_test_pd.id.values
X_test_pd = data_test_pd.drop(['id'], axis=1)

X_test = X_test_pd.values







#%%
#training
rg=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=1000, learning_rate=1.)
rg.fit(X_train, y_train)
gini=testing(y_valid,rg.predict(X_valid))
print('gini score is:')
print(gini)



#%%
#sending data to CSV
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)

p = rg.predict(X_test)
sub['target']=p
sub.to_csv('submit.csv', index=False)



