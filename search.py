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
        if azimuth!=0.0 or anisotropy!=1.0:
            rotmat = make_rotation_matrix2D(azimuth,anisotropy)
        else:
            rotmat = None

        KDTree.__init__(self,points,rotmat)
