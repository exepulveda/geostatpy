import numpy as np
import scipy.spatial

from geometry import sqdistance, make_rotation_matrix

class KDTree(object):
    def __init__(self,points,azimuth=0.0,dip=0.0,plunge=0.0,anisotropy=[1.0,1.0]):
        #create rotmat
        self._rotmat = make_rotation_matrix([azimuth,dip,plunge],anisotropy)
        #rotate points
        self._rotated_points = np.dot(points,self._rotmat)
        #create kdtree
        self._kdtree = scipy.spatial.cKDTree(self._rotated_points)

    def search(self,points,maxdata=1,max_distance=np.inf):
        rp = np.dot(points,self._rotmat)
        d,i = self._kdtree.query(rp, k=maxdata, distance_upper_bound=max_distance)
        #remove infinity distances
        if isinstance(d, np.ndarray):
            idx = np.where(np.isfinite(d))
            return d[idx],i[idx]
        else:
            return d,i if np.isfinite(d) else None,None

def kriging_system(kriging_type,vm,point,points,data):
    A,b = vm.kriging_system(point,points)
    if kriging_type == "ordinay":            
        #add lagrange
        n = len(A)
        A2 = np.ones((n+1,n+1))
        A2[0:n,0:n] = A
        A2[n,n] = 0.0
        b2 = np.ones((n+1))
        b2[0:n] = b
        A = A2
        b = b2
        
    x = np.linalg.solve(A,b)
    
    if kriging_type == "simple":            
        estimation = mean + np.sum((data - mean) * x)
    else:
        estimation = np.sum(data * x[:-1])

    variance = vm.max_covariance() -np.sum(x*b)

    return estimation,variance,A,b,x
        
def kriging3d_cross_validation(kriging_type,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree(points,azimuth,dip,plunge,anisotropy)
    
    max_covaraince = vm.max_covariance()
    
    n = len(points)
    ret = np.empty((n,2))
    
    errors = np.empty(n)
    non_estimated = 0
    
    if full:
        ret_indices = []
        
    
    for i,point in enumerate(points):
        #serach one more to eliminate itself
        d,indices = kdtree.search(point,maxdata+1,search_range)
        
        #remove first, because it is itself
        d = d[1:]
        indices = indices[1:]
        if full:
            ret_indices += [indices]
        
        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            errors[i] = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,point,points[indices,:],data[indices])
            errors[i] = data[i] - estimation

            #print A,b,x,estimation,variance,points[indices]

        
        ret[i,0] = estimation
        ret[i,1] = variance
        
    err = np.sum(errors**2)/n
    if full:
        return ret,err,non_estimated,errors,ret_indices
    else:
        return ret,err,non_estimated


def kriging3d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree(points,azimuth,dip,plunge,anisotropy)
    
    max_covaraince = vm.max_covariance()
    
    n = len(points)
    ret = np.empty((n,2))
    
    non_estimated = 0
    
    if full:
        ret_indices = []
        
    
    for i,point in enumerate(output_points):
        d,indices = kdtree.search(point,maxdata,search_range)
        
        if full:
            ret_indices += [indices]
        
        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,point,points[indices,:],data[indices])

            #print A,b,x,estimation,variance,points[indices]

        
        ret[i,0] = estimation
        ret[i,1] = variance
        
    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated
