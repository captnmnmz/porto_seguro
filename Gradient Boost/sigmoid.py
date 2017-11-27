from numpy import *
from math import e
from math import pow
import numpy as np
from scipy.special import expit
def sigmoid(z):
    # Computes the sigmoid of z.
    #we use the native sigmoid function in python because the normal function
    #returns runtime overflows (tries to calculte too large values)
    

    return expit(z)
