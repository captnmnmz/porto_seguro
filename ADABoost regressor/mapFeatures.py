import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def mapFeatures(X, degree = 6):
    # Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
    # Note: It works only for two dimensional input features.
    

            
    poly = PolynomialFeatures(degree, include_bias=False)
    F = poly.fit_transform(X)

    return F
