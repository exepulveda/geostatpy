import numpy as np
import scipy.spatial

from geometry import sqdistance, make_rotation_matrix, make_rotation_matrix2D

'''Search parameter
'''
class SearchParameter(object):
    def __init__(self,mindata,maxdata,range,anisotropy=None,azimuth=0.0,dip=0.0,plunge=0.0):
        self.mindata = mindata
        self.maxdata = maxdata
        self.range = range
        if anisotropy is None:
            anisotropy = np.ones(2)

        self.rotmat = make_rotation_matrix([azimuth,dip,plunge],anisotropy)

'''KDTree to search neigborhs. To support rotated search and anisotropy, the input points are rotated and scaled before the kdtree is built.
Consequentely, all search point are rotated and scaled before.'''
class KDTree(object):
    def __init__(self,points,rotmat=None):
        self._rotmat = rotmat
        #rotate and scale points
        if rotmat is None:
            self._rotated_points = points.copy()
        else:
            self._rotated_points = np.dot(points,self._rotmat)
        #create kdtree
        self._kdtree = scipy.spatial.cKDTree(self._rotated_points)

    def query_ball(self,p,r):
        return self._kdtree.query_ball_point(p,r)

    def search(self,points,maxdata=1,max_distance=np.inf):
        #rotate and scale points
        if self._rotmat is None:
            d,i = self._kdtree.query(points, k=maxdata, distance_upper_bound=max_distance)
        else:
            rp = np.dot(points,self._rotmat)
            d,i = self._kdtree.query(rp, k=maxdata, distance_upper_bound=max_distance)
        #remove infinity distances
        if isinstance(d, np.ndarray):
            idx = np.where(np.isfinite(d))
            return d[idx],i[idx]
        else:
            return d,i if np.isfinite(d) else None

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
        
'''KDTree for 3D points to search neigborhs'''
class FullKDTree3D(KDTree):
    def __init__(self,points,bhid=None,azimuth=0.0,dip=0.0,plunge=0.0,anisotropy=[1.0,1.0]):
        n,m = points.shape
        self.bhid = bhid

        if m != 3:
            raise KrigingSetupException("points are not 3D")

        if len(anisotropy) != 2:
            raise KrigingSetupException("anisotropy must be 2 length array or list")

        #create rotmat
        self._rotmat = make_rotation_matrix([azimuth,dip,plunge],anisotropy)

        KDTree.__init__(self,points,self._rotmat)
        
    def search(self,points,mindata=1,maxdata=10,min_octants=None,min_per_octant=None,max_per_octant=None,max_distance=np.inf):
        #check the number of points
        search_points = np.asarray(points)
        if len(search_points.shape) <= 1:
            search_points = search_points[np.newaxis,:]

        #rotate and scale points
        if self._rotmat is not None:
            search_points = np.dot(search_points,self._rotmat)
            
        n = len(search_points)
        
        ret = []
        #
        idxs = self._kdtree.query_ball_point(search_points, r=max_distance,return_sorted=True)
        for i,idx in enumerate(idxs):
            idx = filter_neigborhs(search_points[i],idx,self._rotated_points,bhid=None,mindata=mindata,maxdata=maxdata,min_octants=min_octants,min_per_octant=min_per_octant,max_per_octant=max_per_octant,max_per_bh=None)
            ret += [idx]

        if len(points.shape) > 1:
            return ret
        else:
            return ret[0]

    def query_ball(self,points,distance=np.inf):
        #check the number of points
        search_points = np.asarray(points)
        if len(search_points.shape) <= 1:
            search_points = search_points[np.newaxis,:]

        #rotate and scale points
        if self._rotmat is not None:
            search_points = np.dot(search_points,self._rotmat)
            
        n = len(search_points)
        
       
        idxs = self._kdtree.query_ball_point(search_points[0,:],r=distance,return_sorted=True)
        distances = np.linalg.norm(search_points[0,:] -self._rotated_points[idxs],axis=1)

        return idxs,distances


'''KDTree for 2D points to search neigborhs'''
class KDTree2D(KDTree):
    def __init__(self,points,azimuth=0.0,anisotropy=1.0):
        n,m = points.shape

        if m != 2:
            raise KrigingSetupException("points are not 2D")

        #create rotmat
        if azimuth!=0.0 or anisotropy!=1.0:
            rotmat = make_rotation_matrix2D(azimuth,anisotropy)
        else:
            rotmat = None

        KDTree.__init__(self,points,rotmat)
        
def filter_neigborhs(loc,indices,locations,bhid=None,mindata=1,maxdata=10,min_octants=None,min_per_octant=None,max_per_octant=None,max_per_bh=None):
    ''''
    indices are sorted in ascending order
    Octant filter has more priority than borehole filter
    '''
    ns = len(indices)
    filter_by_octants = min_octants is not None or min_per_octant is not None or max_per_octant is not None
    if filter_by_octants: 
        #just limit to octants
        octants = [[] for _ in range(8)]
        
        bhids = {}
        
        for idx in indices:
            diffloc = loc - locations[idx,:]
            if diffloc[0] < 0:
                if diffloc[1] < 0:
                    if diffloc[2] < 0:
                        oct = 0
                    else:
                        oct = 1
                else:
                    if diffloc[2] < 0:
                        oct = 2
                    else:
                        oct = 3
            else:
                if diffloc[1] < 0:
                    if diffloc[2] < 0:
                        oct = 4
                    else:
                        oct = 5
                else:
                    if diffloc[2] < 0:
                        oct = 6
                    else:
                        oct = 7
                        
            #check max_per_bh
            if max_per_bh is not None:
                if bhid[idx] not in bhids:
                    bhids[bhid[idx]] = []
                bhids[bhid[idx]] += [idx]
                    
                accept = len(bhid[idx]) < max_per_bh
                
            else:
                accept = True

            if accept:
                #check max octants
                if max_per_octant is None or len(octants[oct]) < max_per_octant:
                    octants[oct] += [idx]
        
        #now check min_per_octant
        if min_per_octant is not None:
            for oct,idx in enumerate(octants):
                 if len(idx) < min_per_octant:
                     octants[oct] = [] #it does not meet the minimum

        #now check min_octants
        if min_octants is not None:
            #count octants not empty octants
            informed_octants = 0
            ret = []
            for oct in octants:
                if len(oct) > 0:
                    informed_octants += 1
                    ret += oct

            if informed_octants >= min_octants:
                return ret
            else:
                return []
        else:
            ret = []
            for oct in octants:
                ret += oct

            return ret
            
    elif not filter_by_octants and max_per_bh is not None: 
        bhids = {}
        for idx in indices:
            if bhid[idx] not in bhids:
                bhids[bhid[idx]] = []

            if len(bhid[idx]) < max_per_bh:
                bhids[bhid[idx]] += [idx]
                accept = True
            else:
                accept = False

        ret  = []
        for k,v in bhids.items():
            ret  += v
            
        return ret

    else:
        #just min and max
        if ns >= mindata:
            ret = indices[:min(ns,maxdata)]
            return ret
        else:
            return []
