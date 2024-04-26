#  data process
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  
# from ..utils.gmr import Gmr, plot_gmm
# from .KMP_functions import *  

refTraj = {} 

t = np.array([0, 1, 2]) 

mu = np.zeros((2,2)) 

refTraj['t'] = t 
refTraj['mu'] = mu

print(refTraj)