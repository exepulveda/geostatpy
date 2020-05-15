import numpy as np

from geometry import sqdistance, make_rotation_matrix, make_rotation_matrix2D
from scipy.special import kv
from scipy.special import gamma

class VariogramDefinitionException(Exception):
    pass


def spherical(h,c,a,**kargs):
    '''Spherica model
    '''
    hr = h/a

    idx = np.where(hr > 1)[0]

    ret = c*(1.0-hr*(1.5-0.5*hr**2))

    if len(idx) > 0:
        if isinstance(ret, np.ndarray):
            ret[idx] = 0.0
        else:
            ret = 0.0

    return ret

def exponential(h,c,a,**kargs):
    '''Exponential model
    '''
    hr = h/a

    ret = c*np.exp(-3.0*hr)

    return ret
    
def gaussian(h,c,a,**kargs):
    hr = h/a
    ret = c*np.exp(-3.0*hr**2)
    return ret

def mater(h,c,a,**kargs):
    hr = h/a
    nu = kargs['nu']
    #return c*(1.0-((2.0**(1.0-nu)/gamma(nu)) * (hr**nu) * kv(nu,hr)))
    return c*(((2.0**(1.0-nu)/gamma(nu)) * (hr**nu) * kv(nu,hr)))

def mater_emery(h,c,a,**kargs):
    hr = h/a
    nu = kargs['nu']
    return c*((hr/2.0**nu)/gamma(nu) * kv(nu,hr))
    

'''Supported variogram models'''
structure_types = { "spherical":spherical, "exponential":exponential, "gaussian":gaussian, "matern":mater}
structure_codes = { "spherical":1, "exponential":2,"gaussian":4,"matern":10}

class VariogramStructure(object):
    def __init__(self,structure_type,sill,ranges,angles,**kargs):
        ranges = np.array(ranges)
        self.sill = sill
        self.structure_type = structure_type
        self._ranges = ranges
        self._range = ranges[0]
        self._function = structure_types[structure_type]
        self._angles = angles
        self._kargs = kargs

    def covariance(self,p1,p2):
        h2 = sqdistance(p1,p2,self._rotmat)
        h = np.sqrt(h2)
        #print('covariance',p1,p2,h2,h)
        return self._function(h,self.sill,self._range,**self._kargs)

    def regularized(self,w,gamma_bar_small_scale,W,gamma_bar_large_scale):
        new_sill = self.sill * (1.0 - gamma_bar_large_scale)/ (1.0-gamma_bar_small_scale)
        new_ranges = self._ranges + (W-w)
        print('self._ranges',self._ranges)
        print('W,w',W,w)
        new_st = self.__class__(self.structure_type,new_sill,new_ranges,self._angles)

        return new_st

class VariogramStructure3D(VariogramStructure):
    def __init__(self,structure_type,sill,ranges,angles=[0.0,0.0,0.0],**kargs):
        VariogramStructure.__init__(self,structure_type,sill,ranges,angles,**kargs)
        self._rotmat = make_rotation_matrix(angles,self._ranges[1:]/self._ranges[0])

class VariogramStructure2D(VariogramStructure):
    def __init__(self,structure_type,sill,ranges,angle=0.0):
        VariogramStructure.__init__(self,structure_type,sill,ranges,angle)
        self._rotmat = make_rotation_matrix2D(angle,ranges[1]/ranges[0])

class VariogramModel(object):
    def __init__(self,nugget=0.0):
        self.nugget = nugget
        self.structures = []
        self._default_rotmat = make_rotation_matrix([0.0,0.0,0.0])
        self._maxcov = None

    def __str__(self):
        str = '\nnugget=%f\n'%(self.nugget)
        for i,st in enumerate(self.structures):
            str += 'structure %d:'%((i+1))
            str += '\ttype: %s'%(st.structure_type)
            str += '\tsill: %f'%(st.sill)
            str += '\tranges: %s'%(st._ranges)
            str += '\tangles: %s'%(st._angles)
            str += '\tparam: %s'%(st._kargs)
            str += '\n'
        return str

    def max_covariance(self):
        if self._maxcov is None:
            self._maxcov = self.nugget + sum([st.sill for st in self.structures])

        return self._maxcov

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
            
    def gamm(self,p1,p2):
        return self.max_covariance() - self.covariance(p1,p2)

    def compute_variogram(self,directions):
        #generate vectors accoring lag and direction
        max_covariance = self.max_covariance()
        ret = []

        origin = np.zeros(3)
        for direction in directions:
            lags,lag_size,azimuth,dip = direction

            #print lags,lag_size,azimuth,dip

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

    def evaluate_variogram(self,azimuth,dip,lags):
        #generate vectors accoring lag and direction
        max_covariance = self.max_covariance()

        origin = np.zeros(3)

        #print lags,lag_size,azimuth,dip

        vector = np.zeros((len(lags),3))

        #lag directional vector
        xoff = np.sin(np.radians(azimuth))*np.cos(np.radians(dip))
        yoff = np.cos(np.radians(azimuth))*np.cos(np.radians(dip))
        zoff = np.sin(np.radians(azimuth))

        vector[:,0] = lags
        vector[:,1] = lags
        vector[:,2] = lags

        vector[:,0] *= xoff
        vector[:,1] *= yoff
        vector[:,2] *= zoff

        covariances = self.covariance(origin,vector)

        h = np.sqrt(np.sum(vector**2,1))
        gam = max_covariance - covariances

        return gam

    def gamma_bar(self,block_size,discretization):
        assert len(block_size) == len(discretization)
        
        nd = len(discretization)
        n = np.product(discretization)
        sizes = np.array(block_size)
        d = sizes / np.array(np.array(discretization)-1)
        #print('discretization=',discretization)
        #print('d=',d)

        if nd == 2:
            ret = np.empty((n,2))

            p = 0
            for i in range(discretization[0]):
                for j in range(discretization[1]):
                    ret[p] = np.array([d[0] * i,d[1] * j])
                    p += 1


        elif nd == 3:
            ret = np.empty((n,3))

            p = 0
            for i in range(discretization[0]):
                for j in range(discretization[1]):
                    for k in range(discretization[2]):
                        ret[p] = np.array([d[0] * i,d[1] * j,d[2] * k])
                        p += 1
        else:
            raise Exception("Only 2D or3D supported")

        #ret = ret +d*0.5
        #ret = ret -0.5*sizes
        
        #print('ret=',ret)
        #quit()

        max_cova = self.max_covariance()

        d2 = 0.0
        for i in range(n):
            p1 = ret[i]
            gamma = max_cova - self.covariance(p1,ret)
            d2 += np.sum(gamma)

        d2 = d2/n**2
        return d2

    def regularized(self,small_sizes,block_sizes,discretization):
        '''return a regularized variogram model.
        '''
        assert len(small_sizes) == len(discretization)
        assert len(block_sizes) == len(discretization)

        gamma_bar_small_scale = self.gamma_bar(small_sizes,discretization)
        gamma_bar_large_scale = self.gamma_bar(block_sizes,discretization)

        print('gamma_bar_small_scale',gamma_bar_small_scale)
        print('gamma_bar_large_scale',gamma_bar_large_scale)
        #w and W for nugget are volumes
        w = np.prod(small_sizes)
        W = np.prod(block_sizes)
        reg_nugget = self.nugget * w / W

        reg_vmodel = self.__class__(reg_nugget)
        #w and W for sill are linear length
        w = small_sizes
        W = block_sizes
        for i,st in enumerate(self.structures):
            reg_vmodel.structures += [st.regularized(w,gamma_bar_small_scale,W,gamma_bar_large_scale)]

        return reg_vmodel

    def kriging_system(self,p,neigborhood,dicretized_points=None):
        #create Ax=b system
        n = len(neigborhood)

        A = np.zeros((n,n))
        for i in range(n):
            p1 = neigborhood[i,:]
            p2 = neigborhood[i:,:]
            cov = self.covariance(p1,p2)
            A[i,i:] = cov

        #symmetric
        for i in range(n):
            A[i+1:,i] = A[i,i+1:]

        if dicretized_points is None:
            b = self.covariance(p,neigborhood)
            #print(p,b)
        else:
            b = np.zeros(n)
            for pd in (dicretized_points + p):
                b += self.covariance(pd,neigborhood)

            b = b / len(dicretized_points)


        #print "System",A,b
        return A,b

    def kriging_system_many(self,points,neigborhood,dicretized_points=None):
        #create Ax=b system
        n = len(neigborhood)
        m = len(points)

        A = np.zeros((n,n))
        for i in range(n):
            p1 = neigborhood[i,:]
            p2 = neigborhood[i:,:]
            cov = self.covariance(p1,p2)
            A[i,i:] = cov

        #symmetric
        for i in range(n):
            A[i+1:,i] = A[i,i+1:]

        B = np.empty((n,m))

        for i,p in enumerate(points):
            if dicretized_points is None:
                b = self.covariance(p,neigborhood)
                #print(p,b)
            else:
                b = np.zeros(n)
                for pd in (dicretized_points + p):
                    b += self.covariance(pd,neigborhood)

                b = b / len(dicretized_points)

            B[:,i] =b

        #print "System",A,b
        return A,B

    def block_gamma(self,dicretized_points):
        n = len(dicretized_points)
        max_covariance = self.max_covariance()
        gamma = np.zeros((n,n))
        for i in range(n):
            p1 = dicretized_points[i,:]
            p2 = dicretized_points[i:,:]
            cov = self.covariance(p1,p2)
            gamma[i,i:] = max_covariance - cov

        #symmetric
        for i in range(n):
            gamma[i+1:,i] = gamma[i,i+1:]

        return gamma

    def block_cova(self,block_size,discretization):
        assert len(block_size) == len(discretization)
        nd = len(discretization)
        n = np.product(discretization)
        sizes = np.array(block_size)
        d = sizes / np.array(discretization)

        if nd == 2:
            ret = np.empty((n,2))

            p = 0
            for i in range(discretization[0]):
                for j in range(discretization[1]):
                    ret[p] = np.array([d[0] * i,d[1] * j])
                    p += 1


        elif nd == 3:
            ret = np.empty((n,3))

            p = 0
            for i in range(discretization[0]):
                for j in range(discretization[1]):
                    for k in range(discretization[2]):
                        ret[p] = np.array([d[0] * i,d[1] * j,d[2] * k])
                        p += 1
        else:
            raise Exception("Only 2D or3D supported")

        ret = ret +d*0.5
        ret = ret -0.5*sizes

        max_cova = self.max_covariance()

        d2 = 0.0
        for i in range(n):
            p1 = ret[i]
            cova = self.covariance(p1,ret)
            d2 += np.sum(cova)

        d2 = d2/n**2
        return d2


class VariogramModel3D(VariogramModel):
    def __init__(self,nugget=0.0):
        VariogramModel.__init__(self,nugget)
        self._default_rotmat = make_rotation_matrix([0.0,0.0,0.0])

    def add_structure(self,structure_type,sill,ranges,angles,**kargs):
        if structure_type not in structure_types:
            raise VariogramDefinitionException("Undefined structure type")

        vs = VariogramStructure3D(structure_type,sill,ranges,angles,**kargs)
        self.structures += [vs]

    def compute_variogram(self,directions):
        #generate vectors according lag and direction
        max_covariance = self.max_covariance()
        ret = []

        origin = np.zeros(3)
        for direction in directions:
            lags,lag_size,azimuth,dip = direction



            vector = np.zeros((lags+1,3))

            #lag directional vector
            xoff = np.sin(np.radians(azimuth))*np.cos(np.radians(dip))*lag_size
            yoff = np.cos(np.radians(azimuth))*np.cos(np.radians(dip))*lag_size
            zoff = np.sin(np.radians(dip))*lag_size

            #print lags,lag_size,azimuth,dip,xoff,yoff,zoff


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

class VariogramModel2D(VariogramModel):
    def __init__(self,nugget=0.0):
        VariogramModel.__init__(self,nugget)
        self._default_rotmat = make_rotation_matrix2D(0.0,0.0)

    def add_structure(self,structure_type,sill,ranges,angle):
        if structure_type not in structure_types:
            raise VariogramDefinitionException("Undefined structure type")

        vs = VariogramStructure2D(structure_type,sill,ranges,angle)
        self.structures += [vs]

    def compute_variogram(self,directions):
        #generate vectors accoring lag and direction
        max_covariance = self.max_covariance()
        ret = []

        origin = np.zeros(2)
        for direction in directions:
            lags,lag_size,azimuth,dip = direction

            vector = np.zeros((lags+1,2))

            #lag directional vector
            xoff = np.sin(np.radians(azimuth))*lag_size
            yoff = np.cos(np.radians(azimuth))*lag_size

            vector[1:,0] = (np.arange(lags)+1)
            vector[1:,1] = vector[1:,0]

            vector[:,0] *= xoff
            vector[:,1] *= yoff

            covariances = self.covariance(origin,vector)

            h = np.sqrt(np.sum(vector**2,1))
            gam = max_covariance - covariances

            vmodel = np.empty((lags+1,2))
            vmodel[:,0] = h
            vmodel[:,1] = gam

            ret += [(vmodel,max_covariance)]

        return ret
