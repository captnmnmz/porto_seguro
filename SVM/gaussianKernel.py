from numpy import *

def gaussianKernel(X1, X2, sigma = 0.1):
    m = X1.shape[0]
    K = zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Gaussian kernel (see the assignment
    #				for more details).
    

    for i in range (m):
        for j in range(K.shape[1]):
            K[i,j] = pow(e,(linalg.norm(X1[i,:]-X2[j,:])**2)/(2*(sigma**2)))
    
    # =============================================================

    return K
