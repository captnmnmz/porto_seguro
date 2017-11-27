from numpy import *
from sigmoid import sigmoid

def computeCostreg(theta, X, y, l):
    # Computes the cost of using theta as the parameter 
    # for logistic regression.
    m = X.shape[0] # number of training examples
    h = sigmoid(dot(X, theta))

    tol = .000000000000001  
    h[h < tol] = tol  # values close to zero are set to tol
    h[(h < 1 + tol) & (h > 1 - tol)] = 1 - tol  # values close to 1 get set to 1 - tol
    
    J = (1./m)*(-dot(y.T,log(h)) - dot((1.0-y).T,log(1.0-h))) + (l / (2.0 * m)) * dot(theta[1:].T,theta[1:])
    
    return J



