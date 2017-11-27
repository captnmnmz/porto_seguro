from numpy import *
from sigmoid import sigmoid

def computeGradreg(theta, X, y, l):
    # Computes the gradient of the cost with respect to the parameters.

    m = X.shape[0] # number of training examples
    
    grad = zeros_like(theta) #initialize gradient

    h = sigmoid(dot(X,theta))

    delta = h - y

    grad[0] =  delta.T.dot(X[:, 0]) # intercept grad
    grad[1:] = (dot(delta.T, X[:,1:]) + l*theta[1:]) # gradients of the remaining features

    return (1.0 / m) *grad
    
