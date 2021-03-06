from numpy import *

def gaussianKernel(X1, X2, sigma = 0.1):
    m = X1.shape[0]
    K = zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Gaussian kernel (see the assignment
    #				for more details).
    

    for i in range(m):
        K[i,:] = np.exp((-(linalg.norm(X1[i,:]-X2, axis=1)**2))/(2*sigma**2))
    
    # =============================================================

    return K
