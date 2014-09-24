import numpy as np

import geometry

def check_angles(dp,azimuth,dip,atol,dtol,horizontal_bandwith,vertical_bandwith):
    max_horizontal_limit = horizontal_bandwith * np.cos(np.radians(atol)) / np.sin(np.radians(atol))
    max_vertical_limit = vertical_bandwith * np.cos(np.radians(dtol)) / np.sin(np.radians(dtol))
    
    ok = np.empty(len(dp),dtype=np.bool)
    ok[:] = False
    
    ds = np.sum(dp,axis=1)
    
    rotmat = geometry.make_rotation_matrix([azimuth,dip,0])
    
    #print "rotmat",rotmat
    
    dp_rotated = np.dot(dp,rotmat)
    
    #print dp_rotated
    #now the vector "is" in the Y axis for easy calculations
    horizontal_size = dp_rotated[:,0]
    vertical_size = dp_rotated[:,2]
    
    #first filter: horizontal_bandwith
    indices = np.where(horizontal_size <= horizontal_bandwith)[0]
    ok[indices] = True
    
    print "1",ok
    #second filter: vertical_bandwith
    indices2 = np.where(vertical_size[indices] <= vertical_bandwith)[0]
    ok[indices[indices2]] = True
    print "2",ok
    
    #third filter: horizontal_angle_tolerance
    indices = indices[indices2]
    horizontal_limit = np.sqrt(ds[indices]+horizontal_size[indices]**2)
    horizontal_angle = np.arctan(dp_rotated[indices,0]/dp_rotated[indices,1])
    
    indices2 = np.where(np.logical_and(horizontal_limit <= max_horizontal_limit,np.abs(horizontal_angle) <= np.radians(atol)))[0]
    ok[indices[indices2]] = True
    print "3",ok

    #forth filter: vertical_angle_tolerance
    indices = indices[indices2]
    vertical_limit = np.sqrt(ds[indices]+vertical_size[indices]**2)
    vectical_angle = np.arctan(dp_rotated[indices,2]/dp_rotated[indices,1])
    
    indices2 = np.where(np.logical_and(vertical_limit <= max_vertical_limit,np.abs(vectical_angle) <= np.radians(dtol)))[0]
    ok[indices[indices2]] = True
    print "4",ok
    
    return ok


def check_azimuth(h,dx,dy,dxs,dys,dxy,uvx_azimuth,uvy_azimuth,csatol,bandwh,uvhdec,uvzdec,csdtol,bandwd,epsilon=1.0e-6):
    n = len(dx)
    
    ret = np.zeros(n,dtype=np.bool)
    
    dcazm = (dx*uvx_azimuth+dy*uvy_azimuth)/dxy

    indices = np.where(dxy<epsilon)[0]
    if len(indices) > 0:
        dcazm[indices] = 1.0
        
    
    #check azimtuh tolerance
    indices = np.where(np.abs(dcazm) >= csatol)[0]

    print csatol,dcazm,indices
    
    if len(indices) == 0:
        return ret

    #check bandwith to remainders
    band = uvx_azimuth*dy[indices] - uvy_azimuth*dx[indices]
    
    indices2 = np.where(np.abs(band) < bandwh)[0]
    
    print "2",bandwh,band,indices2
    
    if len(indices2) == 0:
        return ret
    
    indices = indices[indices2]

    print "3",indices
    
    #DIP
    indices2 = np.where(dcazm[indices] <0)[0]
    if len(indices2) < 0:
        indices3 = indices[indices2]
        dxy[indices3] = -dxy[indices3]


    dcdec = (dxy[indices]*uvhdec+dz[indices]*uvzdec)/h[indices]

    indices2 = np.where(h[indices] <= epsilon)[0]
    dcdec[indices2] = 0.0
    
    indices2 = np.where(np.abs(dcdec) >= csdtol)[0]
    
    print "4",csdtol,dcdec,indices2
    
    if len(indices2) == 0:
        return ret
    
    indices = indices[indices2]

    print "5",indices
    
    #check bandwith to remainders
    band = uvhdec*dz[indices] - uvzdec*dxy[indices]

    indices2 = np.where(np.abs(band) <= bandwd)[0]
    
    print "6",bandwd,band,indices2,dz,dxy

    
    if len(indices2) > 0:
        indices = indices[indices2]
        ret[indices] = True

    return ret

if __name__ == "__main__":
    dip = 0.0
    azm = 45.0
    atol = 5.0
    dtol = 5.0
    bandwh = 0.5
    bandwd = 0.5
    
    x = np.array([1,2,1,2,1,2,3])
    y = np.array([1,1,2,2,3,3,3])
    z = np.array([0,0,0,0,0,0,0])
    
    points = np.array(zip(x,y,z))
    
    print points

    n = len(x)
    
    for i in xrange(1):
        p0 = points[i,:]
        dp = points[(i+1):]-p0
        
        print "checking",p0,points[(i+1):]
        
        ret = check_angles(dp,azm,dip,atol,dtol,bandwh,bandwd)
        
        print ret
