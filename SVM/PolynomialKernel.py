#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:04:47 2017

@author: bastienchevallier
"""

from numpy import *

def PolynomialKernel(X1, X2, degree = 2, c= 1):
    m = X1.shape[0]
    K = zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Gaussian kernel (see the assignment
    #				for more details).
    for i in range(m):
        K[i,:] = pow(dot(X2, X1[i,:]) + c ,degree)
    
    # =============================================================

    return K