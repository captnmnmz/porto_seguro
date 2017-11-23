#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:46:38 2017

@author: bastienchevallier
"""

import numpy as np

def LaplacianKernel(X1, X2):
    m = X1.shape[0]
    K = zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Laplacian kernel (see the assignment
    #				for more details).

    for i in range(m):
        K[i,:] = np.exp((-(linalg.norm(X1[i,:]-X2, axis=1)))/(2*sigma**2))
    
    # =============================================================

    return K
