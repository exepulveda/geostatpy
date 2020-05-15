import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path += ["/home/esepulveda/Documents/projects/geostatpy"]
import decorrelating

# Generate sample data
np.random.seed(0)
n_samples = 20000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal

S = np.c_[s1, s2]

S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
z = np.dot(S, A.T)  # Generate observations

ppg = decorrelating.ProjectionPursuitGaussianizer()
y = ppg.fit_transform(z)

#
print np.mean(y,axis=0)
print np.std(y,axis=0)

#create a sample

simulations = np.random.normal(size=(5000,2))  # Add noise

z2 = ppg.inverse_transform(simulations)

#print "error",np.sum((z-z2)**2)


plt.plot(z[:,0],z[:,1],"+")
plt.plot(z2[:,0],z2[:,1],"*")

plt.show()
