from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from predict import *
from mapFeatures import mapFeatures
from sklearn.preprocessing import StandardScaler
from computeCostreg import computeCostreg
from computeGradreg import computeGradreg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from undersampling import undersampling
from drop import drop
from dummy import dummy
from createMeta import createMeta
from createPoly import createPoly
from selectFeatures import selectFeatures
from testing import testing
from target_encode import target_encode
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
#data_train_pd, selected_vars=selectFeatures(data_train_pd)
#data_test_pd = data_test_pd[selected_vars + ['id']]
#%%
# The first two columns contains the exam scores and the third column contains the label.
# TODO NORMALISATION
y = data_train_pd.target.values
X_pd = data_train_pd.drop(['id','target'], axis=1)
X = X_pd.values
scaler=StandardScaler()
X=scaler.fit_transform(X)
#CREATE CROSS VALIDATION ENSEMBLES
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=None)

test_id = data_test_pd.id.values
X_test_pd = data_test_pd.drop(['id'], axis=1)

X_test = X_test_pd.values
X_test=scaler.transform(X_test)





#%%
##training K FOLD
seed=45
kf = StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)
pred_test_full=0
pred_valid_full=0
cv_score=[]
i=1
for train_index,test_index in kf.split(X_train,y_train):    
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train[train_index],y_train[test_index]
    
    lr = LogisticRegression(class_weight='balanced',C=0.005)
    lr.fit(xtr, ytr)
    pred_test = lr.predict_proba(xvl)[:,1]
    score = roc_auc_score(yvl,pred_test)
    print('roc_auc_score',score)
    cv_score.append(score)
    pred_valid_full += lr.predict_proba(X_valid)[:,1]
    pred_test_full += lr.predict_proba(X_test)[:,1]
    i+=1


y_pred_valid=pred_valid_full/5.
y_pred=pred_test_full/5.

#%%
#GINI

gini=testing( y_valid,y_pred_valid)
print("gini is:")
print(gini)


#%%
#sending data to CSV
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)

p = y_pred
sub['target']=p
sub.to_csv('submit.csv', index=False)



