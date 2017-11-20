#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:46:38 2017

@author: bastienchevallier
"""

import numpy as np

def linearKernel(X1, X2):
    # Computes the linear Kernel between two set of features
    m = X1.shape[0]
    K = np.zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the linear kernel (see the assignment
    #				for more details).
    
    for i in range (m):
        for j in range(K.shape[1]):
            K[i,j]+= np.dot(X1[i,:],X2[j,:])

    # =============================================================
        
    return K
