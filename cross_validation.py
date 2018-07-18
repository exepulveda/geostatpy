import numpy as np
import scipy as sc

import variogram
import h5py

def create_kriging_system_at(x,locations,vm,ordinary=True):
    n,m = locations.shape
    assert (m == 3)
    
    if ordinary:
        n2 = n + 1
    else:
        n2 = n

    a = np.ones((n2,n2))
    for i in range(n):
        pi = locations[i]

        cova = vm.covariance(pi,locations[i:,:])

        a[i,i:n] = cova
        a[i:n,i] = cova

    b = np.ones(n2)
    cova = vm.covariance(x,locations)

    b[:n] = cova


    if ordinary:
        a[-1,-1] = 0

    assert np.all(a == a.T)

    return a,b


def create_kriging_system(locations,vm,ordinary=True):
    n,m = locations.shape
    assert (m == 3)
    
    if ordinary:
        n2 = n + 1
    else:
        n2 = n

    a = np.ones((n2,n2))
    for i in range(n):
        pi = locations[i]

        cova = vm.covariance(pi,locations[i:,:])

        a[i,i:n] = cova
        a[i:n,i] = cova

    if ordinary:
        a[-1,-1] = 0

    assert np.all(a == a.T)

    return a
    
def calculate_inverse(a):
    return np.linalg.inv(a)
    
def estimation_at(z,k,inva,n):
    indices = np.arange(n)
    indices = np.delete(indices,k)
    
    ba0a0 = inva[k,k]
    ba0a = inva[k,indices]

    #print "ba0a0",ba0a0
    #print "ba0a / ba0a0",ba0a / ba0a0
    #print "sum(ba0a / ba0a0)",np.sum(ba0a / ba0a0)
    est = -np.sum(z[indices] * (ba0a / ba0a0))
    var = 1.0/ba0a0
    
    return est,var
        
    
if __name__ == "__main__":
    data = np.loadtxt("samples.csv",delimiter=";")
    n,m = data.shape
    print n,m

    locations = data[:,0:3]
    z = data[:,3]
    vmodel = variogram.VariogramModel3D(0.5)
    vmodel.add_structure("spherical",1.5,[10,10,10],[0,0,0])
    vmodel.add_structure("spherical",1.0,[300,300,300],[0,0,0])

    #using invertion
    a = create_kriging_system(locations,vmodel)
    inva = calculate_inverse(a)
    est,var = estimation_at(z,0,inva,n)
    
    #using the kriging system
    locations = data[1:,0:3]
    z = data[1:,3]
    n = len(z)
    a2 = create_kriging_system(locations,vmodel)
    p =data[0,0:3]
    a2,b2 = create_kriging_system_at(p,locations,vmodel)
    #solve sysmte
    l = np.linalg.solve(a2,b2)
    est2 = np.sum(z*l[:n])
    var2 =  vmodel.covariance(p,p) - np.sum(b2*l)
    print data[0,3],est,var,est2,var2
    quit()
    
    
    inverse_file = "ainv.mat"

    zmean = np.mean(z)
    zcentered = z - zmean

    
    if False:
        fd = h5py.File(inverse_file,"a")
        ds = fd.create_dataset("cova",(n+1,n+1))
        ds[:,:] = a
    
        ds = fd.create_dataset("cova_inv",(n+1,n+1))
        inva = calculate_inverse(a)
        ds[:,:] = inva
        fd.close()

    fd = h5py.File(inverse_file,"r")
    ds = fd["cova_inv"]
    inva = ds[:,:]
    fd.close()
        
    
    
    #print "filling kriging system"
    #a = create_kriging_system(locations,vmodel)
    #print "inverting matrix"
    #inva = calculate_inverse(a)

    for i in range(n):
        est = estimation_at(z,i,inva,n)
        print z[i],est

    

