import numpy as np
import numpy.ma as ma
import math

import pdb

def sqdistance(p1,p2,rotmat=None):
    diff = p1 - p2
    if rotmat is not None:
        diff = np.dot(diff,rotmat.T)

    if len(diff.shape) > 1:
        return np.sum(diff**2,axis=1)
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
    
def rescale_model(finner_grid_data,finner_grid_sizes,target_grid_sizes):
    n,m,p = finner_grid_data.shape
    total_sizes = np.asarray(finner_grid_sizes) * np.asarray(finner_grid_data.shape)
    
    target_nodes = np.ceil(total_sizes/target_grid_sizes)
    
    nx = int(target_nodes[0])
    ny = int(target_nodes[1])
    nz = int(target_nodes[2])
    
    target_sum_data = np.zeros((nx,ny,nz))
    target_count_data = np.zeros((nx,ny,nz))
    
    #ratio of sizes
    sx =  float(target_grid_sizes[0]/finner_grid_sizes[0])
    sy =  float(target_grid_sizes[1]/finner_grid_sizes[1])
    sz =  float(target_grid_sizes[2]/finner_grid_sizes[2])

    #print(nx,ny,nz)
    #print(sx,sy,sz)
    
    for i in range(nx):
        si_f = int(math.floor(i*sx))
        si_c = int(math.ceil(i*sx))
        ei_f = min(int(math.floor((i+1)*sx)),n)
        ei_c = min(int(math.ceil((i+1)*sx)),n)
        #check for imcomplete cells
        #diff_si = si_c - i*sx
        #diff_ei = i*sx
        for j in range(ny):
            sj_f = int(math.floor(j*sy))
            sj_c = int(math.ceil(j*sy))
            ej_f = min(int(math.floor((j+1)*sy)),m)
            ej_c = min(int(math.ceil((j+1)*sy)),m)
            for k in range(nz):
                sk_f = int(math.floor(k*sz))
                sk_c = int(math.ceil(k*sz))
                ek_f = min(int(math.floor((k+1)*sz)),p)
                ek_c = min(int(math.ceil((k+1)*sz)),p)
                #print(i,j,k,si_f,si_c,ei_f,ei_c,sj_f,sj_c,ej_f,ej_c,sk_f,sk_c,ek_f,ek_c)
                patch_complete = finner_grid_data[si_c:ei_f,sj_c:ej_f,sk_c:ek_f]
                target_sum_data[i,j,k] = np.sum(patch_complete)
                target_count_data[i,j,k] = patch_complete.shape[0]*patch_complete.shape[1]*patch_complete.shape[2]
                #check for imcomplete cells
                #sf_c 
                #return None
                
    return target_sum_data / target_count_data

def rescale_model_sim(finner_grid_data,finner_grid_sizes,target_grid_sizes):
    if np.array_equal(finner_grid_sizes,target_grid_sizes):
        return finner_grid_data
    
    n,m,p,nr = finner_grid_data.shape
    total_sizes = np.asarray(finner_grid_sizes) * np.asarray(finner_grid_data.shape[:3])
    
    target_nodes = np.ceil(total_sizes/target_grid_sizes)
    
    nx = int(target_nodes[0])
    ny = int(target_nodes[1])
    nz = int(target_nodes[2])
    
    target_sum_data = np.zeros((nx,ny,nz,nr))
    target_count_data = np.zeros((nx,ny,nz,nr))
    
    #ratio of sizes
    sx =  float(target_grid_sizes[0]/finner_grid_sizes[0])
    sy =  float(target_grid_sizes[1]/finner_grid_sizes[1])
    sz =  float(target_grid_sizes[2]/finner_grid_sizes[2])

    #print(nx,ny,nz)
    #print(sx,sy,sz)
    
    for i in range(nx):
        si_f = int(math.floor(i*sx))
        si_c = int(math.ceil(i*sx))
        ei_f = min(int(math.floor((i+1)*sx)),n)
        ei_c = min(int(math.ceil((i+1)*sx)),n)
        #check for imcomplete cells
        #diff_si = si_c - i*sx
        #diff_ei = i*sx
        for j in range(ny):
            sj_f = int(math.floor(j*sy))
            sj_c = int(math.ceil(j*sy))
            ej_f = min(int(math.floor((j+1)*sy)),m)
            ej_c = min(int(math.ceil((j+1)*sy)),m)
            for k in range(nz):
                sk_f = int(math.floor(k*sz))
                sk_c = int(math.ceil(k*sz))
                ek_f = min(int(math.floor((k+1)*sz)),p)
                ek_c = min(int(math.ceil((k+1)*sz)),p)
                #print(i,j,k,si_f,si_c,ei_f,ei_c,sj_f,sj_c,ej_f,ej_c,sk_f,sk_c,ek_f,ek_c)
                patch_complete = finner_grid_data[si_c:ei_f,sj_c:ej_f,sk_c:ek_f,:]
                target_sum_data[i,j,k,:] = np.sum(patch_complete,axis=(0,1,2))
                target_count_data[i,j,k,:] = patch_complete.shape[0]*patch_complete.shape[1]*patch_complete.shape[2]
                #check for imcomplete cells
                #sf_c 
                #return None
                
    return target_sum_data / target_count_data


#@jit('f8[:,:,:,:](f8[:,:,:],i4[:],f8[:],f8[:])')
def rescale_model_sim_nogrid(finner_grid_data,shape,finner_grid_sizes,target_grid_sizes):
    if np.array_equal(finner_grid_sizes,target_grid_sizes):
        return finner_grid_data
    
    n,nr,nd = finner_grid_data.shape

    #ratio of sizes
    sx =  float(target_grid_sizes[0]/finner_grid_sizes[0])
    sy =  float(target_grid_sizes[1]/finner_grid_sizes[1])
    sz =  float(target_grid_sizes[2]/finner_grid_sizes[2])

    ngx,ngy,ngz = shape
    nx = int(shape[0] / sx)
    ny = int(shape[1] / sy)
    nz = int(shape[2] / sz)
    
    target_data = np.zeros((n,nx,ny,nz,nr))
    #target_count_data = np.zeros((n,nx,ny,nz,nr))
    
    for idx in range(n):
        finner_data = np.array(finner_grid_data[idx,:,:])
        finner_data = finner_data.reshape((nr,ngx,ngy,ngz))
    
        for i in range(nx):
            si_f = int(math.floor(i*sx))
            si_c = int(math.ceil(i*sx))
            ei_f = min(int(math.floor((i+1)*sx)),ngx)
            ei_c = min(int(math.ceil((i+1)*sx)),ngx)
            #check for imcomplete cells
            #diff_si = si_c - i*sx
            #diff_ei = i*sx
            for j in range(ny):
                sj_f = int(math.floor(j*sy))
                sj_c = int(math.ceil(j*sy))
                ej_f = min(int(math.floor((j+1)*sy)),ngy)
                ej_c = min(int(math.ceil((j+1)*sy)),ngy)
                for k in range(nz):
                    sk_f = int(math.floor(k*sz))
                    sk_c = int(math.ceil(k*sz))
                    ek_f = min(int(math.floor((k+1)*sz)),ngz)
                    ek_c = min(int(math.ceil((k+1)*sz)),ngz)
                    
                    #pdb.set_trace()
                    #print(i,j,k,si_f,si_c,ei_f,ei_c,sj_f,sj_c,ej_f,ej_c,sk_f,sk_c,ek_f,ek_c)
                    patch_complete = finner_data[:,si_c:ei_f,sj_c:ej_f,sk_c:ek_f]
                    #pdb.set_trace()
                    target_data[idx,i,j,k,:] = np.mean(patch_complete,axis=(1,2,3))
                    #target_count_data[idx,i,j,k,:] = patch_complete.shape[0]*patch_complete.shape[1]*patch_complete.shape[2]
                    #check for imcomplete cells
                    #sf_c 
                    #return None
                
    return target_data


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

class AbstractGrid(object):
    def __init__(self,nodes,sizes,starts):
        self.nodes = np.array(nodes,dtype=int)
        self.sizes = np.array(sizes,dtype=float)
        self.starts = np.array(starts,dtype=float)
        self.n = np.product(nodes)

    def __len__(self):
        return self.n

    def volume(self):
        return np.prod(self.sizes)

    def subdivide(self):
        '''This method subdivide the grid'''
        new_nodes = self.nodes * 2
        new_sizes = self.sizes / 2.0
        new_starts = (self.starts - self.sizes / 2.0) + new_sizes/2.0
        new_grid = self.__class__(new_nodes,new_sizes,new_starts)
        return new_grid

    def get_indices(self,location):
        '''This method places samples on the grid averaging if it is needed'''
        indices = [int(x) for x in np.floor((location - self.starts) / self.sizes)]
        return indices

    def place_samples_on(self,locations,data,trace=False):
        '''This method places samples on the grid averaging if it is needed'''
        gvalues = np.zeros(tuple(self.nodes))
        gcount = np.zeros(tuple(self.nodes))

        locations_at_origin = locations - (self.starts - self.sizes/2.0)
        for i,loc in enumerate(locations_at_origin):
            indices = [int(x) for x in np.floor(loc / self.sizes)]
            #print(i,loc,indices,locations[i])

            check_limits = [x >= 0 and x < self.nodes[j] for j,x in enumerate(indices)]
            if np.all(check_limits):
                if trace: print(i,loc,indices,data[i],locations[i])
                gvalues[tuple(indices)] += data[i]
                gcount[tuple(indices)] += 1.0

        #print(gvalues)
        #print(gcount)

        non_zero = ma.masked_array(gvalues,mask=(gcount<=0))
        non_zero[:,:] = non_zero / gcount

        return non_zero

    def filter_in(self,*args,locations=None):
        '''This method filter locations that belongs to a specific cell in the grid. We assume that the upper limit can belong
        to the cell, cell 0 also include the lower limit.
        Return the indices of locations that belongs to the cell indices in args
        '''
        ndim = len(args)
        
        assert(locations is not None)
        assert(len(locations.shape) >= 2)
        
        cell_centre = self.get_location(*args)
        min_locations = cell_centre - self.sizes*0.5
        max_locations = cell_centre + self.sizes*0.5
        
        ret = []
        for i,loc in enumerate(locations):
            inside = True
            for j in range(ndim):
                if args[j] == 0: #if the index is the first, the lower limit belongs to the cell
                    if not (min_locations[j] <= loc[j] <= max_locations[j]):
                        inside = False
                        break
                else:
                    if not (min_locations[j] < loc[j] <= max_locations[j]):
                        inside = False
                        break
            if inside:
                ret += [i]
        
        return ret

    def __str__(self):
        return "nodes: %s. sizes=%s. starts=%s"%(list(self.nodes),list(self.sizes),list(self.starts))

class Grid3D(AbstractGrid):
    def __init__(self,nodes,sizes,starts):
        AbstractGrid.__init__(self,nodes,sizes,starts)

        self.nx = self.nodes[0]
        self.nxy = self.nx * self.nodes[1]

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
        return self.get_location(i,j,k)

    def get_location(self,i,j,k):
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
        
    def cover(self,i=0):
        #return 8 grids that cover the original grid
        nx,ny,nz = self.nodes
        
        n = nx//2
        m = ny//2
        p = nz//2
        #first grid covers from 1:n,1:m, second n+1:nx,1:m,etc...
        g1 = Grid3D((n,m,p),self.sizes,self.starts)
        g2 = Grid3D((nx-n,m,p),self.sizes,self.starts + np.array([n*self.sizes[0],0.0,0.0]))
        g3 = Grid3D((n,ny-m,p),self.sizes,self.starts + np.array([0.0,m*self.sizes[1],0.0]))
        g4 = Grid3D((nx-n,ny-m,p),self.sizes,self.starts + np.array([n*self.sizes[0],m*self.sizes[1],0.0]))
        g5 = Grid3D((n,m,nz-p),self.sizes,self.starts)
        g6 = Grid3D((nx-n,m,nz-p),self.sizes,self.starts + np.array([n*self.sizes[0],0.0,p*self.sizes[2]]))
        g7 = Grid3D((n,ny-m,nz-p),self.sizes,self.starts + np.array([0.0,m*self.sizes[1],p*self.sizes[2]]))
        g8 = Grid3D((nx-n,ny-m,nz-p),self.sizes,self.starts + np.array([n*self.sizes[0],m*self.sizes[1],p*self.sizes[2]]))
        
        return ((g1,0,n,0,m,0,p), (g2,n,nx,0,m,0,p), (g3,0,n,m,ny,0,p), (g4,n,nx,m,ny,0,p),
                (g5,0,n,0,m,p,nz),(g6,n,nx,0,m,p,nz),(g7,0,n,m,ny,p,nz),(g8,n,nx,m,ny,p,nz))

    def validate(self):
        #ids
        for k,cell in enumerate(self):
            assert k < self.n
            #reconstruction
            i,j,m = self.indices(k)
            assert k == self.blockindex(i,j,m)

    def on_grid(self,locations,data):
        '''This method locate the samples on the grid'''
        return None

class Grid2D(AbstractGrid):
    def __init__(self,nodes,sizes,starts):
        AbstractGrid.__init__(self,nodes,sizes,starts)
        self.nx = np.product(nodes[0])

    def get_location(self,i,j):
        p = np.array([i,j])
        p = p*self.sizes + self.starts
        return p

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

    def discretize(self,discretization):
        n = np.product(discretization)

        d = self.sizes / np.array(discretization)

        ret = np.empty((n,2))

        p = 0
        for i in range(discretization[0]):
            for j in range(discretization[1]):
                ret[p] = np.array([d[0] * i,d[1] * j])
                p += 1

        ret = ret +d*0.5

        ret = ret -0.5*self.sizes

        return ret
        
    def cover(self):
        #return 4 grids that cover the original grid
        nx,ny = self.nodes
        
        n = nx//2
        m = ny//2
        #first grid covers from 1:n,1:m, second n+1:nx,1:m,etc...
        g1 = Grid2D((n,m),self.sizes,self.starts)
        g2 = Grid2D((nx-n,m),self.sizes,self.starts + np.array([n*self.sizes[0],0.0]))
        g3 = Grid2D((n,ny-m),self.sizes,self.starts + np.array([0.0,m*self.sizes[1]]))
        g4 = Grid2D((nx-n,ny-m),self.sizes,self.starts + np.array([n*self.sizes[0],m*self.sizes[1]]))
        
        return (g1,0,n,0,m),(g2,n,nx,0,m),(g3,0,n,m,ny),(g4,n,nx,m,ny)

    def validate(self):
        #ids
        for k,cell in enumerate(self):
            assert k < self.n
            #reconstruction
            i,j = self.indices(k)
            assert k == self.blockindex(i,j)
