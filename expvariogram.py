import numpy as np
import math
import itertools

def projection(azimuth,dip=0):
    azm = np.radians(-azimuth)
    dip = np.radians(90.0-dip)
   
    rotation_matrix = np.zeros((3,3))
    rotation_matrix[0,0] = math.sin(azm)
    rotation_matrix[1,1] = math.cos(azm)
    rotation_matrix[2,2] = 1.0
    
    return rotation_matrix
    rotation_matrix[0,0] = b
    rotation_matrix[0,1] = a
    rotation_matrix[0,2] = -sin_rad[1]
    rotation_matrix[1,0] = (-cos_rad[2]*sin_rad[0] + sin_rad[2]*sin_rad[1]*cos_rad[0])
    rotation_matrix[1,1] = (cos_rad[2]*cos_rad[0] + sin_rad[2]*sin_rad[1]*sin_rad[0])
    rotation_matrix[1,2] = (sin_rad[2]*cos_rad[1])
    rotation_matrix[2,0] = (sin_rad[2]*sin_rad[0] + cos_rad[2]*sin_rad[1]*cos_rad[0])
    rotation_matrix[2,1] = (-sin_rad[2]*cos_rad[0] + cos_rad[2]*sin_rad[1]*sin_rad[0])
    rotation_matrix[2,2] = (cos_rad[2]*cos_rad[1])

def projectz(ang):
    rang = math.radians(ang)
    rotation_matrix = np.zeros((3,3))

    rotation_matrix[0,0] = math.cos(rang)
    rotation_matrix[0,1] = -math.sin(rang)
    rotation_matrix[1,0] = math.sin(rang)
    rotation_matrix[1,1] = math.cos(rang)
    rotation_matrix[2,2] = 1.0

    return rotation_matrix

def projecty(ang):
    rang = math.radians(ang)
    rotation_matrix = np.zeros((3,3))

    rotation_matrix[0,0] = math.cos(rang)
    rotation_matrix[2,0] = -math.sin(rang)
    rotation_matrix[0,2] = math.sin(rang)
    rotation_matrix[2,2] = math.cos(rang)
    rotation_matrix[1,1] = 1.0
    
    return rotation_matrix
    

def projectx(ang):
    rang = math.radians(ang)
    rotation_matrix = np.zeros((3,3))

    rotation_matrix[1,1] = math.cos(rang)
    rotation_matrix[1,2] = -math.sin(rang)
    rotation_matrix[2,1] = math.sin(rang)
    rotation_matrix[2,2] = math.cos(rang)
    rotation_matrix[0,0] = 1.0

    return rotation_matrix

def projectxz(azimuth,dip):
    rotation_matrix_xy = projectx(dip)
    rotation_matrix_z = projectz(azimuth)
        
    return np.dot(rotation_matrix_xy,rotation_matrix_z)
    
def check_direction(direction,bw,angletol):
    #check bw
    ysize = direction[:,1]
    bw_ret = ysize <= bw
    
    angletol_ret = np.empty(len(direction),dtype=np.bool)
    angletol_ret[:] = False
    #check angle tolerance
    idx = np.where(direction[:,0] != 0)[0]
    if len(idx) >0:
        angle = np.arctan(direction[idx,1]/direction[idx    ,0])
        angletol_ret[idx] = (np.abs(angle) <= np.radians(angletol))
        #print("dir",direction)
        #print("angle",angle,np.radians(angletol))
        #print("bw_ret",bw_ret)
        #print("angletol_ret",angletol_ret)
        #quit()
    
    return bw_ret & angletol_ret
    

def calculate_experimental_variogram_direction(refindex,direction,data,azimuth,azimuth_tolerance,horizontal_bw,dip,dip_tolerance,vertical_bw,results,lags_start,lags_end):
    #project into x axis
    rotmat_x_axis = projectxz(0,0)
    rotmat_direction = projectxz(90-azimuth,dip)
    
    rotmat = np.dot(rotmat_x_axis,rotmat_direction)
    
    #print("rotmat",rotmat)
    #print("dir",direction)
    
    vectors = np.dot(direction,rotmat)
    #now vectors are rotated to X axis
    v_azimuth = np.arctan(vectors[:,1]/vectors[:,0])
    v_dip = np.arctan(vectors[:,1]/vectors[:,0])
    
    #print("vectors",vectors)
    #quit()
    
    ret = check_direction(vectors,horizontal_bw,azimuth_tolerance)
    
    print(refindex,ret)
    
    idx = np.where(ret)[0]
    if len(idx) > 0:
        gam = (data[refindex] - data[refindex+idx]) ** 2
        distance = np.sqrt(np.sum(vectors[idx,:]**2,1))
        
        for i,d in enumerate(distance):
            #results,lags_start,lags_end
            lower = np.searchsorted(lags_start, d)
            upper = np.searchsorted(lags_end, d,side='right')
            print(i,d,lower,upper)
            for j in range(lower-1,upper+1):
                print("lag into",j,d,lags_start[j],lags_end[j])
                if d >= lags_start[j] and d <= lags_end[j]:
                    results[j,0] += 1
                    results[j,1] += d
                    results[j,2] += gam[i]*0.5
                    print("add into lag into",j,d,gam[i]*0.5,refindex,idx[i]+refindex)
                #print(j,d)
            
    #return gam

def calculate_experimental_variogram(points,data,lagdirs):
    n = len(points)
    
    #prepare results allocation
    results = []
    lags_starts = []
    lags_ends = []
    for lagdir in lagdirs:
        lags,lag_size,lag_tolerance,azimuth,azimuth_tolerance,horizontal_bw,dip,dip_tolerance,vertical_bw = lagdir
        #create lag return storage
        results += [np.zeros((lags,4))]
        lags_starts += [np.arange(1,lags+1)*lag_size - lag_tolerance]
        lags_ends += [np.arange(1,lags+1)*lag_size + lag_tolerance]
        
    
    for i in range(n):
        pi = points[i,:]
        pj = points[i:,:]
        diff_vector = np.abs(pi - pj)

        for j,lagdir in enumerate(lagdirs):
            lags,lag_size,lag_tolerance,azimuth,azimuth_tolerance,horizontal_bw,dip,dip_tolerance,vertical_bw = lagdir

            ret = calculate_experimental_variogram_direction(i,diff_vector,data,azimuth,azimuth_tolerance,horizontal_bw,dip,dip_tolerance,vertical_bw,results[j],lags_starts[j],lags_ends[j])
            #print ret
        
    for i in range(len(lagdirs)):
        results[i][:,2] = results[i][:,2] / results[i][:,0]
        results[i][:,1] = results[i][:,1] / results[i][:,0]
            
    print(results)

if __name__ == "__main__":
    data = np.array([
            [1.0, 1.0, 1.0, 0.12, 0, 4, 10],
            [1.0, 2.0, 1.0, 0.13, 0.028, 4, 10],
            [2.0, 1.0, 1.0, 0.13, 0.027, 4, 10],
            [2.0, 2.0, 1.0, 0.19, 0, 4, 10],
            [2.0, 3.0, 1.0, 0.19, 0.039, 4, 10]
        ])
        
    points = data[:,0:3]
    cut = data[:,3]

    ret = calculate_experimental_variogram(points,cut,[(4,1.0,0.5,45.0,44.0,1.1,45.0,44.0,1.1)])
