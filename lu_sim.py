import numpy as np
import numpy.ma as ma
from scipy.linalg import lu, cholesky
from numpy.linalg import inv

def fill_cova(loc1,loc2,vmodel):
    n,d1 = loc1.shape
    if loc2 is not None:
        m,d2 = loc2.shape
        assert d1 == d2
    else:
        m = n

    cova = np.empty((n,m))

    for i,p1 in enumerate(loc1):
        if loc2 is None:
            cova[i,i:]  = vmodel.covariance(p1,loc1[i:,:])
            cova[i+1:,i] = cova[i,i+1:]
        else:
            cova[i,:]  = vmodel.covariance(p1,loc2)


    return cova

def simulate_lu_decom(sim_locations,sample_locations,vmodel):
    c11 = fill_cova(sample_locations,None,vmodel)
    c21 = fill_cova(sim_locations,sample_locations,vmodel)
    c22 = fill_cova(sim_locations,None,vmodel)

    u11 = cholesky(c11)
    l11 = u11.T
    u11_inv = inv(u11)

    l21 = c21 @ u11_inv
    u12 = l21.T

    l22 = cholesky(c22-l21@u12,lower=True)

    return u11_inv.T,l21,l22



    l11,u11 = lu(c11,permute_l= True)

    l11_inv = inv(l11)
    a21t = l11_inv @ c21.T
    a21 = a21t.T
    b12 = a21t

    l22,u22 = lu(c22-l21@u12,permute_l= True)

    return a21,l11_inv,l22

def simulate_lu(nrealizations,sim_locations,sample_locations,sample_values,vmodel):
    m = len(sim_locations)

    l11_inv,l21,l22 = simulate_lu_decom(sim_locations,sample_locations,vmodel)

    w2 = np.random.normal(size=(m,nrealizations))

    cond_part = l21 @ l11_inv @ sample_values

    cond_sim = np.empty(((m,nrealizations)))
    for i in range(nrealizations):
        cond_sim[:,i] = cond_part + l22 @ w2[:,i]

    return cond_sim
