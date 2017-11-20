from numpy import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from pylab import scatter, show, legend, xlabel, ylabel, contour, title, plot
import numpy as np

# # Load the dataset
# Uncomment the following command if you want to load the microchip dataset
data = loadtxt('microchipdata.txt', delimiter=',')

# Uncomment the following command in order to load bus dataset
# data = genfromtxt('bus.csv',delimiter=',')

X = data[:, 0:2] 
y = data[:, 2]
Xval = dataeval[:, 0:2]
yval = dataeval[:, 2]

# # Plot data 
plt.plot(X[:,0][y == 1], X[:,1][y == 1], 'r+', label="c1")
plt.plot(X[:,0][y == 0], X[:,1][y == 0], 'bo', label="c2")
plt.legend(['y = 1', 'y = 0'],numpoints=1)
plt.show()

sigma = [0.01,0.03,0.1,0.3,1,3] # Gaussian kernel variance
C = [0.01,0.03,0.1,0.3,1,3]  # SVM regularization parameter
Accuracy = zeros((len(sigma),len(C)))

for l in range(len(sigma)):
    for n in range (len(C)):
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
        svc = SVC(C = C[n], kernel="precomputed")
        svc.fit(gaussianKernel(X,X,sigma[i]),y)
        
        # Plot the decision boundary
        u = linspace(min(X[:, 0]),max(X[:, 0]), 200)
        v = linspace(min(X[:, 1]),max(X[:, 1]), 200)
        z = zeros(shape=(len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = svc.predict(gaussianKernel(array([[u[i],v[j]]]),X,sigma[i]))
                
        plot(X[:,0][y == 1], X[:,1][y == 1], 'r+', label="c1")
        plot(X[:,0][y == 0], X[:,1][y == 0], 'bo', label="c2")
        contour(u, v, z.T, [0])
        legend(['y = 1', 'y = 0', 'Decision boundary'],numpoints=1)
        show()
        
        #Compute accuracy on the validation set
        p = svc.predict(gaussianKernel(Xval,Xval,sigma))
        counter = 0
        for i in range(yval.size):
            if p[i] == yval[i]:
                counter += 1
                
        Accuracy[i][j]= (counter / float(y.size) * 100.0)

print(Accuracy)
print ("Best case")

