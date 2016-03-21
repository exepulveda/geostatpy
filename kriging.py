import numpy as np
import scipy.spatial

from geometry import sqdistance, make_rotation_matrix, make_rotation_matrix2D

supported_kriging_types = set(["simple","ordinary"])

class KrigingSetupException(Exception):
    pass

'''KDTree to search neigborhs. To support rotated search and anisotropy, the input points are rotated and scaled before the kdtree is built.
Consequentely, all search point are rotated and scaled before.'''
class KDTree(object):
    def __init__(self,points,rotmat):
        self._rotmat = rotmat
        #rotate and scale points
        self._rotated_points = np.dot(points,self._rotmat)
        #create kdtree
        self._kdtree = scipy.spatial.cKDTree(self._rotated_points)

    def search(self,points,maxdata=1,max_distance=np.inf):
        #rotate and scale points
        rp = np.dot(points,self._rotmat)
        d,i = self._kdtree.query(rp, k=maxdata, distance_upper_bound=max_distance)
        #remove infinity distances
        if isinstance(d, np.ndarray):
            idx = np.where(np.isfinite(d))
            return d[idx],i[idx]
        else:
            return d,i if np.isfinite(d) else None,None

'''KDTree for 3D points to search neigborhs'''
class KDTree3D(KDTree):
    def __init__(self,points,azimuth=0.0,dip=0.0,plunge=0.0,anisotropy=[1.0,1.0]):
        n,m = points.shape
        
        if m != 3:
            raise KrigingSetupException("points are not 3D")

        if len(anisotropy) != 2:
            raise KrigingSetupException("anisotropy must be 2 length array or list")
        
        #create rotmat
        rotmat = make_rotation_matrix([azimuth,dip,plunge],anisotropy)
        
        KDTree.__init__(self,points,rotmat.T)

'''KDTree for 2D points to search neigborhs'''
class KDTree2D(KDTree):
    def __init__(self,points,azimuth=0.0,anisotropy=1.0):
        n,m = points.shape
        
        if m != 2:
            raise KrigingSetupException("points are not 2D")

        #create rotmat
        rotmat = make_rotation_matrix2D(azimuth,anisotropy)
        
        KDTree.__init__(self,points,rotmat)

def kriging_system(kriging_type,vm,max_variance,point,points,data,dicretized_points=None):
    A,b = vm.kriging_system(point,points,dicretized_points)
    if kriging_type == "ordinary":
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
    
    #print A.shape,b.shape,x.shape
    
    if kriging_type == "simple":            
        estimation = mean + np.sum((data - mean) * x)
    else:
        estimation = np.sum(data * x[:-1])

    variance = max_variance -np.sum(x*b)

    return estimation,variance,A,b,x

'''Abstract cross validation by kriging'''
def kriging_cross_validation(kriging_type,points,data,vm,mean,mindata,maxdata,search_range,kdtree,full=False):
    max_covariance = vm.max_covariance()
    
    n = len(points)
    ret = np.empty((n,2))
    
    errors = np.empty(n)
    non_estimated = 0
    
    if full:
        ret_indices = []
        
    
    for i,point in enumerate(points):
        #serach one more to eliminate itself
        d,indices = kdtree.search(point,maxdata+1,search_range)
        
        #remove zero distance
        zind = np.where(d>= 0.00001)[0]
        if len(zind) < n:
            d =  d[zind]
            indices = indices[zind]

        if full:
            ret_indices += [indices]
        
        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            errors[i] = np.nan
            non_estimated +=1
        else:
            try:
                estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices])
                errors[i] = data[i] - estimation
            except:
                estimation = np.nan
                variance = np.nan
                errors[i] = np.nan
                non_estimated +=1
                print i,point,"kriging problem",d,i

            #print A,b,x,estimation,variance

        
        ret[i,0] = estimation
        ret[i,1] = variance
        
    err = np.sum(errors**2)/n
    if full:
        return ret,err,non_estimated,errors,ret_indices
    else:
        return ret,err,non_estimated
        
'''Calculate cross validation by kriging'''
def kriging3d_cross_validation(kriging_type,points,data,vm,mean=None,mindata=1,maxdata=10,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)
    
    ret = kriging_cross_validation(kriging_type,points,data,vm,mean,mindata,maxdata,search_range,kdtree,full)
    
    return ret

'''Calculate cross validation by kriging for 2D points'''
def kriging2d_cross_validation(kriging_type,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,search_range=np.inf,anisotropy=1.0,full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree2D(points,azimuth,anisotropy)

    ret = kriging_cross_validation(kriging_type,points,data,vm,mean,mindata,maxdata,search_range,kdtree,full)
    
    return ret

'''Calculate puntual kriging for 3D points'''
def kriging3d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)

    max_covariance = vm.max_covariance()
    
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
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices])

            #print A,b,x,estimation,variance,points[indices]

        
        ret[i,0] = estimation
        ret[i,1] = variance
        
    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

def kriging2d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,search_range=np.inf,anisotropy=1.0,full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree2D(points,azimuth,anisotropy)
    
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

'''Calculate puntual kriging for 3D points'''
def kriging3d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)
    
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

'''Calculate block kriging for 3D points'''
def kriging3d_block(kriging_type,grid,points,data,vm,discritization=None,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)
    
    if discritization is not None:
        dblock = grid.discretize(discritization)
        npd = len(dblock)
        max_covariance = 0
        for dp in dblock:
            max_covariance += np.sum(vm.covariance(dp,dblock))
            
        max_covariance -= vm.nugget*npd
            
        max_covariance /= npd**2
    else:
        dblock = None
        npd = 1
        max_covariance = vm.max_covariance()
    
    #print "max_covariance",max_covariance
    
    n = len(grid)
    ret = np.empty((n,2))
    
    non_estimated = 0
    
    if full:
        ret_indices = []
            
    for i,point in enumerate(grid):
        #print i,point
        d,indices = kdtree.search(point,maxdata,search_range)
        
        if full:
            ret_indices += [indices]
        
        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices],dblock)

            #print A,b,x,estimation,variance,points[indices]

        
        ret[i,0] = estimation
        ret[i,1] = variance
        
    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

'''Calculate block kriging for 3D points'''
def kriging3d_block_iterator(kriging_type,grid,points,data,vm,discretization=None,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)
    
    if discretization is not None:
        dblock = grid.discretize(discretization)
        npd = len(dblock)
        max_covariance = 0
        for dp in dblock:
            max_covariance += np.sum(vm.covariance(dp,dblock))
            
        max_covariance -= vm.nugget*npd
            
        max_covariance /= npd**2
    else:
        dblock = None
        npd = 1
        max_covariance = vm.max_covariance()
    
    #print "max_covariance",max_covariance
    
    for i,point in enumerate(grid):
        d,indices = kdtree.search(point,maxdata,search_range)
        
        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices],dblock)

        
        if full:
            yield point,estimation,variance,indices
        else:
            yield point,estimation,variance
