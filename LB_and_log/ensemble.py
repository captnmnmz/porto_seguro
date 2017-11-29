# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:15:37 2017

@author: jules
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from gini_comp import gini_xgb
import xgboost as xgb



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models, xgb_params):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
        self.xgb_params = xgb_params

    def fit_predict(self, _X, _y, valid, target_valid, T):
        X = np.array(_X)
        y = np.array(_y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)+1))
        S_test = np.zeros((T.shape[0], len(self.base_models)+1))
        S_valid = np.zeros((valid.shape[0], len(self.base_models)+1))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))
            S_valid_i = np.zeros((valid.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]   
                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
                S_valid_i[:,j] = clf.predict_proba(valid)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)
            S_valid[:,i] = S_valid_i.mean(axis=1)
            
            

        #xgboost
        watchlist = [(xgb.DMatrix(_X, _y), 'train'), (xgb.DMatrix(valid, target_valid), 'valid')]



        print("creating XGB model")

        xgb_model = xgb.train(self.xgb_params, xgb.DMatrix(_X, _y), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=70)
        y_pred = xgb_model.predict(xgb.DMatrix(_X), ntree_limit=xgb_model.best_ntree_limit)
        S_train[:,3] = y_pred


            
        S_test[:,3]=xgb_model.predict(xgb.DMatrix(T.values), ntree_limit=xgb_model.best_ntree_limit)
        S_valid[:,3]=xgb_model.predict(xgb.DMatrix(valid), ntree_limit=xgb_model.best_ntree_limit)
        
        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        res2 = self.stacker.predict_proba(S_valid)[:,1]
        return res, res2