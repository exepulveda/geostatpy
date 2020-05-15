import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

from sklearn.decomposition import FastICA, PCA


'''
Test ICA
'''

file_name = "/home/esepulveda/Documents/projects/geostatpy/muestras.csv"
database = pd.read_csv(file_name,na_values=[-999,""])

n = len(database)

X = np.empty((n,2))
X[:,0] = database["cut"]
X[:,1] = database["au"]

var1 = X[:,0]
var2 = X[:,1]

print np.corrcoef(var1,var2)


ica = FastICA(whiten=False,max_iter=100000000,algorithm="parallel")#,fun="cube")
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix


'''
[[ 0.4929941   0.87003265]
 [-0.87003265  0.4929941 ]]

'''

print np.mean(S_,axis=0)
print A_

database["cut_ica"] = S_[:,0]
database["au_ica"] = S_[:,1]

var1 = S_[:,0]
var2 = S_[:,1]

print np.corrcoef(var1,var2)
