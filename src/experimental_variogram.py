import numpy as np
import os.path
import time
import subprocess

import geometry

GSLIB_PATH = "~/gslib90"

def calculate_variogram(filename,param_filename,output_filename,lag,directions,tmin=-np.inf,tmax=np.inf,standardize_sill=False):
    lines = create_gamv_parameter(filename,output_filename,lag,directions,tmin,tmax,standardize_sill)
    
    fd = open(param_filename,"w")
    fd.write('\n'.join(lines))
    fd.close()

    return run_gslib("gamv",param_filename)
    
def run_gslib(program,parameter_filename="gamv.par",executable_path=GSLIB_PATH):
    executable_path = os.path.expanduser(executable_path)
    
    cmd = "{0} {1}".format(os.path.join(executable_path,program),parameter_filename)
    
    print "executing...",cmd
    
    t1 = time.time()
    
    #call
    try:
        ret = subprocess.call([os.path.join(executable_path,program), parameter_filename])
        status = 0
        msg = ""
    except error as CalledProcessError:
        status = error.returncode
        msg =  error.output
    
    t2 = time.time()
    
    return (status,msg,t2-t1)

def create_gamv_parameter(filename,output_filename,lag,directions,tmin=-np.inf,tmax=np.inf,standardize_sill=False):
    print "directions",directions
    
    lags,lag_size,lag_tolerance = lag
    
    lines = []
    lines += ["Parameters for GAMV"]
    lines += ["*******************"]
    lines += [""]
    lines += ["START OF PARAMETERS:"]
    lines += ["{0}".format(filename)]
    lines += ["1 2 3"]
    lines += ["2 4 5"]
    lines += ["{0}    {1}".format(tmin,tmax)]
    lines += ["{0}".format(output_filename)]
    lines += ["{0}".format(lags)]
    lines += ["{0}".format(lag_size)]
    lines += ["{0}".format(lag_tolerance)]
    lines += ["{0}".format(len(directions))]
    for direction in directions:
        azm,atol,bandwh,dip,dtol,bandwd = direction
        lines += ["{0} {1} {2} {3} {4} {5}".format(azm,atol,bandwh,dip,dtol,bandwd)]
    lines += ["{0}".format(1 if standardize_sill else 0)]
    lines += ["1"]
    lines += ["1 1 1"]
        
    return lines

def check_lags(h,lags,lag_size,lag_tolerance=None):
    n = len(h)
    ret = np.empty((n,2),dtype=np.int)
    ret[:,:] = -1
    
    
    if lag_tolerance is None:
        lag_tolerance = lag_size * 0.5
        
    max_lag_distance = lags*lag_size + lag_tolerance
    
    #print "max_lag_distance",max_lag_distance
    
    indices = np.where(h <= max_lag_distance)[0]
    
    selection = h[indices]
    
    inferior = np.clip(np.ceil((selection-lag_tolerance)/lag_size)-1,0,n-1)
    superior = np.clip(np.floor((selection+lag_tolerance)/lag_size)-1,0,n-1)
    
    print "selection",selection,inferior,superior
    
    ret[indices,0] = inferior
    ret[indices,1] = superior
    
    return ret


def check_angles(dp,ds,azimuth,dip,atol,dtol,horizontal_bandwith,vertical_bandwith,k,points):
    max_horizontal_limit = horizontal_bandwith * np.cos(np.radians(atol)) / np.sin(np.radians(atol))
    max_vertical_limit = vertical_bandwith * np.cos(np.radians(dtol)) / np.sin(np.radians(dtol))
    
    ok = np.empty(len(dp),dtype=np.bool)
    ok[:] = True
    
    rotmat = geometry.make_rotation_matrix([azimuth-90,dip,0])
    
    print "rotmat",rotmat
    print "dp",dp
    
    dp_rotated = np.dot(rotmat,dp.T).T
    
    print "dp_rotated",dp_rotated
    #now the vector "is" in the Y axis for easy calculations
    horizontal_size = dp_rotated[:,0]
    vertical_size = dp_rotated[:,2]
    
    #print "vertical_size",vertical_size
    
    #first filter: horizontal_bandwith
    indices = np.where(np.abs(horizontal_size) > horizontal_bandwith)[0]
    #report why
    for i in indices:
        print "pair",k,i,"was rejected by horizontal bw",horizontal_size[i],horizontal_bandwith
        
    ok[indices] = False
    
    #print "1",ok,horizontal_size,horizontal_bandwith
    
    #second filter: vertical_bandwith
    indices = np.where(ok)[0]
    if len(indices) == 0: #quick return
        return ok
    
    indices2 = np.where(np.abs(vertical_size[indices]) > vertical_bandwith)[0]
    #report why
    for i in indices2:
        print "pair",k,indices[i],"was rejected by vertical bw",vertical_size[indices[i]],vertical_bandwith

    ok[indices[indices2]] = False
    #print "2",ok,vertical_size,vertical_bandwith
    
    #third filter: horizontal_angle_tolerance
    indices = np.where(ok)[0]
    if len(indices) == 0: #quick return
        return ok

    horizontal_limit = np.sqrt(ds[indices]+horizontal_size[indices]**2)
    horizontal_angle = np.arctan(dp_rotated[indices,0]/dp_rotated[indices,1])
    
    indices2 = np.where(np.logical_and(horizontal_limit <= max_horizontal_limit,np.abs(horizontal_angle) > np.radians(atol)))[0]
    ok[indices[indices2]] = False
    #report why
    for i in indices2:
        print "pair",k,indices[i],"was rejected by horizontal angle tolerance",np.degrees(horizontal_angle[i]),atol

    #print "3",ok,np.degrees(horizontal_angle),atol,horizontal_limit,max_horizontal_limit

    #forth filter: vertical_angle_tolerance
    indices = np.where(ok)[0]
    if len(indices) == 0: #quick return
        return ok
    vertical_limit = np.sqrt(ds[indices]+vertical_size[indices]**2)
    vectical_angle = np.arctan(dp_rotated[indices,2]/dp_rotated[indices,1])
    
    indices2 = np.where(np.logical_and(vertical_limit <= max_vertical_limit,np.abs(vectical_angle) > np.radians(dtol)))[0]
    #report why
    for i in indices2:
        print "pair",k,indices[i],"was rejected by vertical angle tolerance",np.degrees(vectical_angle[i]),dtol


    ok[indices[indices2]] = False
    print "4",ok,np.degrees(vectical_angle),dtol,max_vertical_limit,vertical_limit
    
    return ok

def variogram(points,head_values, tail_values,direction,lag):
    azm,atol,bandwh,dip,dtol,bandwd = direction
    lags,lag_size,lag_tolerance = lag
    
    n = len(points)
    
    variogram_result = np.zeros((lags,8))
    
    for i in xrange(n):
        p0 = points[i,:]
        dp = points[(i+1):]-p0
        
        print "processing",i,(i+1),len(dp),p0,dp
        
        ds = np.sum(dp**2,axis=1)
        
        #print "checking lags",len(ds)
        
        #ret = check_angles(dp,ds,azm,dip,atol,dtol,bandwh,bandwd)
        h = np.sqrt(ds)
        #print "checking",p0,points[(i+1):],h
        ret_lags = check_lags(h,lags,lag_size,lag_tolerance)
        
        print "ret_lags",ret_lags
        
        #find ok checked lags
        ok = (ret_lags[:,0] <= ret_lags[:,1])
        
        #print "ok lags",len(ok)
        
        #check angles
        indices = np.where(ok)[0]
        print "indices before angle",indices
        ret_angles = check_angles(dp[indices],ds[indices],azm,dip,atol,dtol,bandwh,bandwd,i,points)
        print "ret_angles",ret_angles
        
        indices = indices[np.where(ret_angles)[0]]
        
        #indices has pairs between point-i and others
        #ret has lags where cummulate variogram calculations
        hvalues = head_values[indices]
        m = len(indices)
        
        for i,index in enumerate(indices):
            sl = slice(ret_lags[index,0],ret_lags[index,1]+1)
            print "cumm",i,index,sl
            
            variogram_result[sl,0] += 1 #count
            variogram_result[sl,1] += h[index] #distance
            variogram_result[sl,2] += (h[index] - head_values[i])**2 #gam

    return variogram_result

if __name__ == "__main__":
    azm = 0.0
    atol = 5.0
    bandwh = 0.2
    dip = 90.0
    dtol = 5.0
    bandwd = 0.2

    directions = [(azm,atol,bandwh,dip,dtol,bandwd)]
    lag = (4,1.0,0.5)
    
    data = np.array([
            [1.0,1.0,1.0,0.12,0,4,10],
            [2.0,1.0,1.0,0.13,0.028,4,10],
            [3.0,1.0,1.0,0.12,0,4,10],
            [1.0,2.0,1.0,0.12,0,4,10],
            [2.0,2.0,1.0,0.13,0.028,4,10],
            [3.0,2.0,1.0,0.13,0.028,4,10],
            [1.0,3.0,1.0,0.12,0,4,10],
            [2.0,3.0,1.0,0.13,0.028,4,10],
            [3.0,3.0,1.0,0.13,0.028,4,10],
            [1.0,1.0,2.0,0.12,0,4,10],
            [2.0,1.0,2.0,0.13,0.028,4,10],
            [3.0,1.0,2.0,0.12,0,4,10],
            [1.0,2.0,2.0,0.12,0,4,10],
            [2.0,2.0,2.0,0.13,0.028,4,10],
            [3.0,2.0,2.0,0.13,0.028,4,10],
            [1.0,3.0,2.0,0.12,0,4,10],
            [2.0,3.0,2.0,0.13,0.028,4,10],
            [3.0,3.0,2.0,0.13,0.028,4,10],
        ])

    values = data[:,3]
    
    points = data[:,:3]
    
    #ret = variogram(points,values, values,direction,lag)    
    
    filename = "muestras.dat"
    param_filename = "gamv.par"
    output_filename = "gamv.out"

    status,msg,ellapsed_time = calculate_variogram(filename,param_filename,output_filename,lag,directions,tmin=0,tmax=1e10,standardize_sill=False)
    
    print status
