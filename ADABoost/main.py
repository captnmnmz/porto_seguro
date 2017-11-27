from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from predict import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
from undersampling import undersampling
from drop import drop
from dummy import dummy
from testing import testing
from createMeta import createMeta
from createPoly import createPoly
from ADA_train import ADA_train
from target_encode import target_encode
from selectFeatures import selectFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

#%%
#useful constants
MAX_ROUNDS = 400
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50  
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
#smooth the values in pas_car_11_cat
train_encoded, test_encoded = target_encode(data_train_pd["ps_car_11_cat"], 
                             data_test_pd["ps_car_11_cat"], 
                             target=data_train_pd.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
data_train_pd['ps_car_11_cat_te'] = train_encoded
data_train_pd.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
data_test_pd['ps_car_11_cat_te'] = test_encoded
data_test_pd.drop('ps_car_11_cat', axis=1, inplace=True)
meta_test.loc['ps_car_11_cat','keep'] = False  # Updating the meta
    
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
# TODO NORMALISATION
y = data_train_pd.target.values
y=2*y-1
X_pd = data_train_pd.drop(['id','target'], axis=1)
X = X_pd.values

#CREATE CROSS VALIDATION ENSEMBLES
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=None)

test_id = data_test_pd.id.values
X_test_pd = data_test_pd.drop(['id'], axis=1)

X_test = X_test_pd.values

#%%
#train model
clf = ADA_train(X_train, y_train, 800)
test_error=clf.score(X_valid, y_valid)
print('after training, the harsh score is :')
print(test_error)

#%%
#GINI
gini=testing(clf.predict_proba(X_valid),y_valid)
print(gini)
#%%Predict values for TEST
y_pred=clf.predict_proba(X_test)




#%%
#sending data to CSV
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)

sub['target']=y_pred
sub.to_csv('submit.csv', index=False)



