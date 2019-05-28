import sys
import gaussian

from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.stats
import mdp

import ppd

def sigma(X):
    return np.corrcoef(X)

def kl_normal(data):
    a,b = data.shape
    
    print a,b
    
    n = np.sqrt(len(data))
    
    ret = 0.0
    for i in range(b):    
        x = (np.mean(data[:,i])-data[:,i])/np.std(data[:,i])
        
        
        pr,b = np.histogram(x,max(n,100),density=True)
        
        #print pr,b
        
        mp = (b[1:] + b[:-1]) / 2.0 #middle points
        
        #r = b[1] - b[0]
        
        
        #print mp
        
        for p,m in zip(pr,mp):
            #pdf
            if p > 0:
                #q1 = norm.cdf(g1)
                #q2 = norm.cdf(g2)
                q = norm.pdf(m)
                #print p,q
                
                if q > 0:
                    ret += p*np.log(p/q)
        
    return ret

def kl(p,q):
    #return np.sum(np.where(q != 0, p * np.log(p / q), 0))  

    a = [(i,j) for i,j in zip(p, q) if j == 0.0 or i == 0.0]
    print a
    
    return - sum([i * np.log(i/j) for i,j in zip(p, q) if j != 0.0 and i != 0])  
    
    '''
\displaystyle \text{KL}(p, q) = 0.5 [\log(\text{det}(\Sigma_2)/\text{det}(\Sigma_1)) + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)'\Sigma_2^{-1}(\mu_2-\mu_1) - N]'''

class AbstractICA(object):
    def __init__(self):
        self._transformed_data = None
        self._ica = None
        
    def fit(self,X):
        self._ica.train(X)
        
        self._transformed_data = self._ica.execute(X)
        
    def fit_transform(self,X):
        self.fit(X)
        return self._transformed_data
        
    def inverse_transform(self,Y):
        return self._ica.inverse(Y)

class JADEICA(AbstractICA):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._ica = mdp.nodes.JADENode() #TDSEPNode() #JADENode() #FastICANode()

class TDSEPICA(AbstractICA):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._ica = mdp.nodes.TDSEPNode() #JADENode() #FastICANode()

class FastICA2(AbstractICA):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._ica = mdp.nodes.FastICANode() #CuBICANode
        
class CuBICA(AbstractICA):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._ica = mdp.nodes.CuBICANode()

class MarginalGaussianizer(object):
    def __init__(self):
        self._transformed_data = None
        self._table = None
        
    def fit(self,X,precision):
        self._dim = len(X.shape)
        self._precision = precision
        
        if len(X.shape) > 1:
            self._dim = X.shape[1]
            self._transformed_data = np.empty_like(X)
            self._table =  []
            for i in range(self._dim):
                td, t = gaussian.marginal_gaussianization(X[:,i],precision=precision)                
                self._transformed_data[:,i] = td
                self._table +=  [t]
        else:
            self._dim = 1
            self._transformed_data, self._table =  gaussian.marginal_gaussianization(X,precision=precision)
        
    def fit_transform(self,X,precision=1000):
        self.fit(X,precision)
        return self._transformed_data
        
    def inverse_transform(self,Y):
        x_lin = norm.cdf(Y)
        x2 = gaussian.inv_marginal_uniformization(x_lin,self._table,self._precision);        
        
        return x2

class ProjectionPursuitGaussianizer(object):
    def __init__(self):
        self._transformed_data = None
        self._pca = None
        #self._rotator = self._methods[method]
        self._steps = []

    def fit(self,X,max_iter=50,ep=1e-3):
        #nscores each dimension
        n,m = X.shape
        
        self._nscore = gaussian.NormalScoreTransformer()
        norm_data = self._nscore.fit_transform(X)
        
        #normalize data
        #self._means = np.mean(X,axis=0)
        #self._stddev = np.std(X,axis=0)
        #norm_data = (X-self._means)/self._stddev

        self._pca = PCA(whiten=False)
        
        z = self._pca.fit_transform(norm_data) #sphere(z) #pca.fit_transform(z) 
        #z = X.copy()
        #print np.cov(z.T)
        
        for i in range(max_iter):
            #a,idx = ppd.find_best_direction(z)
            
            #
            a,idx = ppd.find_best_direction_ga(z)
            print "index at",i,idx,a
            
            #a3,idx3 = ppd.find_best_direction_ts(z)
            
            #print a,idx,a2,idx2,a3,idx3
            
            z,U,nst = ppd.remove_structure(z,a)

            self._steps += [(a,U,nst)]

            #print "U matrix at direction",i,U
            corr_trans = np.corrcoef(z.T)
            print "corr_trans",corr_trans

            
            if idx < ep:
                break

        
        self._transformed_data = z

    def fit_transform(self,X,max_iter=50,ep=1e-3):
        self.fit(X,max_iter,ep)
        return self._transformed_data

    def transform(self,Y):
        z = Y.copy()
        for i in range(len(self._steps)):
            a,U,nst = self._steps[i]
            print "step",i,a.shape,U.shape
            xU = np.dot(z,U)
            
            xU[:,0] = nst.inverse_transform(xU[:,0])
            
            z = np.dot(xU,U.T)
        #
        z = self._pca.inverse_transform(z)
        #desnormailize
        #z = z*self._stddev +  self._means
        z = self._nscore.inverse_transform(z)
        return z

    def inverse_transform(self,Y):
        z = Y.copy()
        print z.shape
        for i in reversed(range(len(self._steps))):
            a,U,nst = self._steps[i]
            print "step",i,a.shape,U.shape
            xU = np.dot(z,U)
            
            xU[:,0] = nst.inverse_transform(xU[:,0])
            
            z = np.dot(xU,U.T)
        #
        z = self._pca.inverse_transform(z)
        #desnormailize
        #z = z*self._stddev +  self._means
        z = self._nscore.inverse_transform(z)
        return z

class Gaussianizer(object):
    _methods = {"pca": PCA, "fastica": FastICA, "jadeica": JADEICA, "tdsep":TDSEPICA, "cubica":CuBICA}
    
    def __init__(self,method="pca"):
        self._transformed_data = None
        self._table = None
        self._rotator = self._methods[method]
        
    def fit(self,X,max_iter=10,ep=1e-5):
        n,m = X.shape
        
        self._ica_transformers = []
        self._ns_transformers = []
        
        transformed_data = np.copy(X)

        #print "Gaussianizer.before loop",np.min(transformed_data),np.max(transformed_data)

        for k in range(max_iter):        
            rot = self._rotator()
            #rot = FastICA(max_iter=2000)
            
            self._ica_transformers += [rot]
            
            #rot = PCA()
            #print "Gaussianizer.fit",k,np.min(transformed_data),np.max(transformed_data)
            
            transformed_data = rot.fit_transform(transformed_data)
            
            self._ns_transformers += [[]]
            for i in range(m):
                #nscore to each dimension
                nst = NormalScoreTransformer()
                td = nst.fit_transform(transformed_data[:,i])
                
                #print "marginal transform",k,i,np.min(td),np.max(td)
                transformed_data[:,i] = td
                
                self._ns_transformers[k] += [nst]

                

            distance = ppd.index_multidimensional(transformed_data,m=100)
            print "distance",k,distance,ep
            
            if distance < ep:
                break
            #now generate rotation matrix
            #A = ica.mixing_
                   
            #transformed_data = np.dot(transformed_data,A)
            
        print "iterations",len(self._ica_transformers),distance
        
        self._transformed_data = transformed_data        
        
    def transform(self,X):
        return gaussian.back_normal_scores(X,self._table)   

    def inverse_transform(self,Y):
        transformed_data = np.copy(Y)

        for k in range(len(self._ica_transformers)-1,-1,-1):
            rot = self._ica_transformers[k]
            nst = self._ns_transformers[k]
            

            for i in range(len(nst)):
                #nscore to each dimension
                i = i
                transformed_data[:,i] = nst[i].inverse_transform(transformed_data[:,i])
                
            transformed_data = rot.inverse_transform(transformed_data)
        
        return transformed_data        
        
        
        

    def fit_transform(self,X,max_iter=10,ep=1e-5):
        self.fit(X,max_iter,ep)
        return self._transformed_data
        
        
def toy():
    # Generate sample data
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal

    S = np.c_[s1, s2]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    #S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 0.5], [0.5, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    #plt.plot(X[:,0],X[:,1],"+")
    #plt.show()

    '''
    x = np.arange(100)

    print x

    x2, T = gaussian.marginal_uniformization(x,porc=1.0,precision=1000)

    print T

    print x2

    x3 = gaussian.inv_marginal_uniformization(x2,T,precision=1000)


    print x3

    '''


    #quit()
    ret = []

    for i in range(1,50):

        gaussianiaser = Gaussianizer()

        print "Gaussianizer",np.min(X),np.max(X)

        T = gaussianiaser.fit_transform(X,max_iter=i)

        print T.shape

        #plt.plot(T[:,0],T[:,1],"+")
        #plt.show()

        #inverse

        U = gaussianiaser.inverse_transform(T)

        #plt.plot(U[:,0],U[:,1],"+")
        #plt.show()

        err = np.sum((X-U)**2) / len(X)
        
        ret += [(i,err)]
        
    for i,err in ret:    
        print i,err

if __name__ == "__main__":
    x = np.random.random(size=(1000,2))
    
    print kl_normal(x)
