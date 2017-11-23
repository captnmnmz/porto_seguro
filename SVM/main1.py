from numpy import *
from linearKernel import linearKernel
from sklearn.model_selection import GridSearchCV
from pylab import scatter, show, legend, xlabel, ylabel, contour, title, plot
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

# # Load the dataset
# # The first two columns contains the exam scores and the third column
# # contains the label.
train = pd.read_csv("../../../train.csv")
test = pd.read_csv("../../../test.csv")

print("Training set has %d rows and %d columns\n"%(train.shape[0], train.shape[1]) )
print("Test set has %d rows and %d columns\n"%(test.shape[0], test.shape[1]))

size_training = 100
size_testing = 100


X = train.drop('target', axis=1).values
y = train['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=None)



#Linear Kernel
#linearKernel(X_train,X_train)
#Polynomial Kernel
#PolynomialKernel(X_train,X_train)
#GaussianKernel
#gaussianKernel(X_train,X_train)
#LaplacianKernel
#LaplacianKernel(X_train,X_train)

#========================= TRAINING ==========================#
    
#pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', C=100., gamma = 0.001)
#model = make_pipeline(pca, svc)

print ("Shape of X_train :", X_train.shape)
print ("Shape of X_test :", X_test.shape)
print ("Shape of y_train :", y_train.shape)
print ("Shape of y_test :", y_test.shape)

param_grid = {'svc__C': [1., 5., 10., 50.],
             'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(svc, param_grid)
print("Fit is running . . .\n")
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Test is running . . .\n")
model = grid.best_estimator_
#TODO check if this is model.predict
y_pred = model.predict(X_test)

target_names = train['target'].unique().astype(str).sort()
print(classification_report(y_test, y_pred,target_names=target_names))

print("==== ACCURACY ====")
accuracy_score(y_test, y_pred)

#C = 100.0  # SVM regularization parameter

#svc_linear = SVC(C = C, kernel="linear")
#svc_linear.fit(linearKernel(X,X),y)

#svc_poly = SVC(C = C, kernel="poly")
#svc_poly.fit(PolynomialKernel(X,X),y)

#svc_sigmoid = SVC(C = C, kernel="sigmoid")
#svc_gaussian.fit(gaussianKernel(X,X),y)

#svc_Laplacian = SVC(C = C, kernel=LaplacianKernel)
#svc_Laplacian.fit(LaplacianKernel(X,X),y)


#=========================TEST==========================#
#print("Test is running")
#y_pred_rbf = svc_rbf.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=target_names))
#

