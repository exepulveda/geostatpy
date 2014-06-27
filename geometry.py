import numpy as np

def sqdistance(p1,p2,rotmat=None):
    diff = p1 - p2
    if rotmat is not None:
        diff = np.dot(diff,rotmat)
        
    if isinstance(diff, np.ndarray):
        return np.sum(diff**2,1)
    else:
        return np.sum(diff**2)
    
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
    rotation_matrix[1,0] = (-cos_rad[2]*sin_rad[0] + sin_rad[2]*sin_rad[1]*cos_rad[0]) * ratios[0]
    rotation_matrix[1,1] = (cos_rad[2]*cos_rad[0] + sin_rad[2]*sin_rad[1]*sin_rad[0]) * ratios[0]
    rotation_matrix[1,2] = (sin_rad[2]*cos_rad[1]) * ratios[0]
    rotation_matrix[2,0] = (sin_rad[2]*sin_rad[0] + cos_rad[2]*sin_rad[1]*cos_rad[0]) * ratios[1]
    rotation_matrix[2,1] = (-sin_rad[2]*cos_rad[0] + cos_rad[2]*sin_rad[1]*sin_rad[0]) * ratios[1]
    rotation_matrix[2,2] = (cos_rad[2]*cos_rad[1]) * ratios[1]

    return rotation_matrix
