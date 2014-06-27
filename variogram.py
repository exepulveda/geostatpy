import numpy as np

from geometry import sqdistance, make_rotation_matrix

def spherical(h,c,a):
    hr = h/a
    
    idx = np.where(hr > 1)[0]

    ret = c*(1.0-hr*(1.5-0.5*hr**2))

    if len(idx) > 0:
        if isinstance(ret, np.ndarray):
            ret[idx] = 0.0
        else:
            ret = 0.0
    
    return ret

def exponential(h,c,a):
    hr = h/a
    
    ret = c*np.exp(-3.0*hr)
    
    return ret
    
structure_types = { "spherical":spherical, "exponential":exponential}

class VariogramStructure(object):
    def __init__(self,structure_type,sill,ranges,angles=[0.0,0.0,0.0]):
        ranges = np.array(ranges)
        self.sill = sill
        self.structure_type = structure_type
        self._range = ranges[0]
        self._rotmat = make_rotation_matrix(angles,ranges[1:]/ranges[0])
        self._function = structure_types[structure_type]
        
    def covariance(self,p1,p2):
        h2 = sqdistance(p1,p2,self._rotmat)
        h = np.sqrt(h2)
        return self._function(h,self.sill,self._range)
        
class VariogramModel(object):
    def __init__(self,nugget=0.0):
        self.nugget = nugget
        self.structures = []
        self._default_rotmat = make_rotation_matrix([0.0,0.0,0.0])
        
    def add_structure(self,structure_type,sill,ranges,angles):
        vs = VariogramStructure(structure_type,sill,ranges,angles)
        self.structures += [vs]

    def max_covariance(self):
        return self.nugget + sum([st.sill for st in self.structures])
        
    def get_default_rotmat(self):
        if len(self.structures) > 0:
            return self.structures[0]._rotmat
        
        return self._default_rotmat
        
    def covariance(self,p1,p2,epsilon=1.0e-5):
        h2 = sqdistance(p1,p2,self.get_default_rotmat())

        ret = np.zeros_like(h2)
        
        for st in self.structures:
            covariance = st.covariance(p1,p2)
            ret += covariance

        if isinstance(h2, np.ndarray):
            idx = np.where(h2 < epsilon)[0]
            if len(idx) > 0:
                ret[idx] = self.max_covariance()
                
            return ret
        else:
            if h2 < epsilon:
                return self.max_covariance()
                
            return ret
        
    def compute_variogram(self,directions):
        #generate vectors accoring lag and direction
        max_covariance = self.max_covariance()
        ret = []
        
        origin = np.zeros(3)
        for direction in directions:
            lags,lag_size,azimuth,dip = direction
            
            vector = np.zeros((lags+1,3))
        
            #lag directional vector
            xoff = np.sin(np.radians(azimuth))*np.cos(np.radians(dip))*lag_size
            yoff = np.cos(np.radians(azimuth))*np.cos(np.radians(dip))*lag_size
            zoff = np.sin(np.radians(azimuth))*lag_size
            
            vector[1:,0] = (np.arange(lags)+1)
            vector[1:,1] = vector[1:,0]
            vector[1:,2] = vector[1:,0]
            
            vector[:,0] *= xoff
            vector[:,1] *= yoff
            vector[:,2] *= zoff
            
            covariances = self.covariance(origin,vector)
            
            h = np.sqrt(np.sum(vector**2,1))
            gam = max_covariance - covariances
            
            vmodel = np.empty((lags+1,2))
            vmodel[:,0] = h
            vmodel[:,1] = gam
            
            ret += [(vmodel,max_covariance)]

        return ret
        
    def kriging_system(self,p,neigborhood):
        #create Ax=b system
        n = len(neigborhood)
        
        A = np.zeros((n,n))
        for i in xrange(n):
            p1 = neigborhood[i,:]
            p2 = neigborhood[i:,:]
            cov = self.covariance(p1,p2)
            A[i,i:] = cov

        #symmetric
        for i in xrange(n):
            A[i+1:,i] = A[i,i+1:]
            
        b = self.covariance(p,neigborhood)

        return A,b
        
        
