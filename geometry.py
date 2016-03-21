import numpy as np

def sqdistance(p1,p2,rotmat=None):
    diff = p1 - p2
    if rotmat is not None:
        diff = np.dot(diff,rotmat.T)
        
    if isinstance(diff, np.ndarray):
        return np.sum(diff**2,1)
    else:
        return np.sum(diff**2)

def make_rotation_z(angle):
    rotmat = np.zeros((3,3))
    a = math.radians(angle)
    rotmat[0,0] = math.cos(a)
    rotmat[0,1] = -math.sin(a)
    rotmat[1,1] = rotmat[0,0]
    rotmat[1,0] = -rotmat[0,1]
    rotmat[2,2] = 1.0
    
    return rotmat

def make_rotation_x(angle):
    rotmat = np.zeros((3,3))
    a = math.radians(angle)
    rotmat[1,1] = math.cos(a)
    rotmat[1,2] = -math.sin(a)
    rotmat[2,2] = rotmat[1,1]
    rotmat[2,1] = -rotmat[1,2]
    rotmat[0,0] = 1.0
    
    return rotmat
    
    
def make_rotation_matrix(angles,ratios=[1.0, 1.0]):
    ''' angles is a 3d vector in degrees: azimuth, dip, plunge
    ratios is a 2d vector with anisotropy
    '''
    rad_angles = np.empty(3)
    if angles[0] >= 0 and angles[0] < 270:
        rad_angles[0] = np.radians(90.0 - angles[0]) 
    else:
        rad_angles[0] = np.radians(450.0 - angles[0]) 
        
    rad_angles[1] = -np.radians(angles[1])
    rad_angles[2] = np.radians(angles[2])
    
    sin_rad = np.sin(rad_angles)
    cos_rad = np.cos(rad_angles)

    rotation_matrix = np.empty((3,3))
    
    rotation_matrix[0,0] = cos_rad[1] * cos_rad[0]
    rotation_matrix[0,1] = cos_rad[1] * sin_rad[0]
    rotation_matrix[0,2] = -sin_rad[1]
    rotation_matrix[1,0] = (-cos_rad[2]*sin_rad[0] + sin_rad[2]*sin_rad[1]*cos_rad[0]) / ratios[0]
    rotation_matrix[1,1] = (cos_rad[2]*cos_rad[0] + sin_rad[2]*sin_rad[1]*sin_rad[0]) / ratios[0]
    rotation_matrix[1,2] = (sin_rad[2]*cos_rad[1]) / ratios[0]
    rotation_matrix[2,0] = (sin_rad[2]*sin_rad[0] + cos_rad[2]*sin_rad[1]*cos_rad[0]) / ratios[1]
    rotation_matrix[2,1] = (-sin_rad[2]*cos_rad[0] + cos_rad[2]*sin_rad[1]*sin_rad[0]) / ratios[1]
    rotation_matrix[2,2] = (cos_rad[2]*cos_rad[1]) / ratios[1]

    return rotation_matrix

def make_rotation_matrix2D(azimuth,ratio=1.0):
    if azimuth >= 0 and azimuth < 270:
        rad_angle = np.radians(90.0 - azimuth) 
    else:
        rad_angle = np.radians(450.0 - azimuth) 
        
    sin_rad = np.sin(rad_angle)
    cos_rad = np.cos(rad_angle)

    rotation_matrix = np.empty((2,2))
    
    rotation_matrix[0,0] = cos_rad
    rotation_matrix[0,1] = sin_rad
    rotation_matrix[1,0] = -sin_rad * ratio
    rotation_matrix[1,1] = cos_rad * ratio

    return rotation_matrix


class GridIterator(object):
    def __init__(self,grid):
        self._grid = grid
        self._current = 0

    def next(self):
        return self.__next__()

    def __next__(self): # Python 3: def __next__(self)
        if self._current >= self._grid.n:
            raise StopIteration
        else:
            self._current += 1
            return self._grid[self._current - 1]
            
class Grid3D(object):
    def __init__(self,nodes,sizes,starts):
        self.nodes = np.array(nodes,dtype=int)
        self.sizes = np.array(sizes,dtype=float)
        self.starts = np.array(starts,dtype=float)
        self.n = np.product(nodes)
        self.nx = self.nodes[0]
        self.nxy = self.nx * self.nodes[1]
        
    def __len__(self):
        return self.n

    def volume(self):
        return np.prod(self.sizes)
        
    def blockindex(self,i,j,k):
        #return i*self.nodes[0] + j*self.nodes[1] * self.nx  + k*self.nodes[2] * self.nxy
        return i + j*self.nx  + k*self.nxy

    def indices(self,blockid):
        k = blockid // self.nxy
        j = (blockid - k * self.nxy) // self.nx
        i = blockid - k * self.nxy - j * self.nx
        return i,j,k

    def __iter__(self):
        return GridIterator(self)

    def __getitem__(self,item): 
        i,j,k = self.indices(item)
        #print item,i,j,k
        p = np.array([i,j,k])
        p = p*self.sizes + self.starts
        return p
        
    def discretize(self,discretization):
        n = np.product(discretization)
        
        d = self.sizes / np.array(discretization)
        
        ret = np.empty((n,3))
        
        p = 0
        for i in range(discretization[0]):
            for j in range(discretization[1]):
                for k in range(discretization[2]):
                    ret[p] = np.array([d[0] * i,d[1] * j,d[2] * k])
                    p += 1
            
        ret = ret +d*0.5

        ret = ret -0.5*self.sizes
        
        return ret
        
    def validate(self):
        #ids
        for k,cell in enumerate(self):
            assert k < self.n
            #reconstruction
            i,j,m = self.indices(k)
            assert k == self.blockindex(i,j,m)
            
class Grid2D(object):
    def __init__(self,nodes,sizes,starts):
        self.nodes = np.array(nodes)
        self.sizes = np.array(sizes)
        self.starts = np.array(starts)
        self.n = np.product(nodes)
        self.nx = np.product(nodes[0])

    def blockindex(self,i,j):
        #return i*self.nodes[0] + j*self.nodes[1] * self.nx
        return i + j*self.nx

    def indices(self,blockid):
        j = blockid // self.nx
        i = blockid - j * self.nx
        return i,j

    def __iter__(self):
        return GridIterator(self)

    def __getitem__(self,item): 
        i,j = self.indices(item)
        p = np.array([i,j])
        p = p*self.sizes + self.starts
        return p

    def get_box(self,item): 
        i,j = self.indices(item)
        pcentre = np.array([i,j])
        pcentre = pcentre*self.sizes + self.starts
        
        
        p1 = pcentre - self.sizes/2.0
        p2 = pcentre + self.sizes/2.0
        
        return p1,p2
        
    def validate(self):
        #ids
        for k,cell in enumerate(self):
            assert k < self.n
            #reconstruction
            i,j = self.indices(k)
            assert k == self.blockindex(i,j)
        
