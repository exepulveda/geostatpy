import numpy as np
import scipy.spatial

from geometry import sqdistance, make_rotation_matrix, make_rotation_matrix2D
from search import KDTree3D,KDTree2D,SearchParameter

supported_kriging_types = set(["simple","ordinary"])

class KrigingSetupException(Exception):
    pass

def kriging_system(kriging_type,vm,max_variance,point,points,data,dicretized_points=None):
    A,b = vm.kriging_system(point,points,dicretized_points)
    if kriging_type == "ordinary":
        #add lagrange
        n = len(A)
        A2 = np.ones((n+1,n+1))
        A2[0:n,0:n] = A
        A2[n,n] = 0.0
        b2 = np.ones((n+1))
        b2[0:n] = b
        A = A2
        b = b2

    x = np.linalg.solve(A,b)

    #print A.shape,b.shape,x.shape

    if kriging_type == "simple":
        estimation = mean + np.sum((data - mean) * x)
    else:
        estimation = np.sum(data * x[:-1])

    variance = max_variance -np.sum(x*b)

    return estimation,variance,A,b,x

'''Abstract cross validation by kriging'''
def kriging_cross_validation(kriging_type,points,data,vm,mean,mindata,maxdata,search_range,kdtree,full=False):
    max_covariance = vm.max_covariance()

    n = len(points)
    ret = np.empty((n,2))

    errors = np.empty(n)
    non_estimated = 0

    if full:
        ret_indices = []


    for i,point in enumerate(points):
        #serach one more to eliminate itself
        d,indices = kdtree.search(point,maxdata+1,search_range)

        #remove zero distance
        zind = np.where(d>= 0.00001)[0]
        if len(zind) < n:
            d =  d[zind]
            indices = indices[zind]

        if full:
            ret_indices += [indices]

        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            errors[i] = np.nan
            non_estimated +=1
        else:
            try:
                estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices])
                errors[i] = data[i] - estimation
            except:
                estimation = np.nan
                variance = np.nan
                errors[i] = np.nan
                non_estimated +=1
                print(i,point,"kriging problem",d,i)

            #print A,b,x,estimation,variance


        ret[i,0] = estimation
        ret[i,1] = variance

    err = np.sum(errors**2)/n
    if full:
        return ret,err,non_estimated,errors,ret_indices
    else:
        return ret,err,non_estimated

'''Calculate cross validation by kriging'''
def kriging3d_cross_validation(kriging_type,points,data,vm,mean=None,mindata=1,maxdata=10,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)

    ret = kriging_cross_validation(kriging_type,points,data,vm,mean,mindata,maxdata,search_range,kdtree,full)

    return ret

'''Calculate cross validation by kriging for 2D points'''
def kriging2d_cross_validation(kriging_type,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,search_range=np.inf,anisotropy=1.0,full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree2D(points,azimuth,anisotropy)

    ret = kriging_cross_validation(kriging_type,points,data,vm,mean,mindata,maxdata,search_range,kdtree,full)

    return ret

'''Calculate puntual kriging for 3D points'''
def kriging3d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)

    max_covariance = vm.max_covariance()

    n = len(points)
    ret = np.empty((n,2))

    non_estimated = 0

    if full:
        ret_indices = []


    for i,point in enumerate(output_points):
        d,indices = kdtree.search(point,maxdata,search_range)

        if full:
            ret_indices += [indices]

        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices])

            #print A,b,x,estimation,variance,points[indices]


        ret[i,0] = estimation
        ret[i,1] = variance

    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

def kriging2d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,search_range=np.inf,anisotropy=1.0,full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree2D(points,azimuth,anisotropy)

    max_covaraince = vm.max_covariance()

    n = len(points)
    ret = np.empty((n,2))

    non_estimated = 0

    if full:
        ret_indices = []


    for i,point in enumerate(output_points):
        d,indices = kdtree.search(point,maxdata,search_range)

        if full:
            ret_indices += [indices]

        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,point,points[indices,:],data[indices])

            #print A,b,x,estimation,variance,points[indices]


        ret[i,0] = estimation
        ret[i,1] = variance

    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

'''Calculate puntual kriging for 3D points'''
def kriging3d_puntual(kriging_type,output_points,points,data,vm,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)

    max_covaraince = vm.max_covariance()

    n = len(points)
    ret = np.empty((n,2))

    non_estimated = 0

    if full:
        ret_indices = []


    for i,point in enumerate(output_points):
        d,indices = kdtree.search(point,maxdata,search_range)

        if full:
            ret_indices += [indices]

        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,point,points[indices,:],data[indices])

            #print A,b,x,estimation,variance,points[indices]


        ret[i,0] = estimation
        ret[i,1] = variance

    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

'''Calculate block kriging for 3D points'''
def kriging3d_block(kriging_type,grid,points,data,vm,discritization=None,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False,debug=1):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)

    if discritization is not None:
        dblock = grid.discretize(discritization)
        npd = len(dblock)
        max_covariance = 0
        for dp in dblock:
            max_covariance += np.sum(vm.covariance(dp,dblock))

        max_covariance -= vm.nugget*npd

        max_covariance /= npd**2
    else:
        dblock = None
        npd = 1
        max_covariance = vm.max_covariance()

    print("block covariance",max_covariance)

    n = len(grid)
    ret = np.empty((n,2))

    non_estimated = 0

    if full:
        ret_indices = []

    for i,point in enumerate(grid):
        #print i,point
        d,indices = kdtree.search(point,maxdata,search_range)
        if debug >=1:
            print('est at:',point,'neigborhs len:',len(indices))
        if full:
            ret_indices += [indices]

        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
            non_estimated +=1
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices],dblock)
        if debug >=3:
            print("ponits:",points[indices])
            print("A=",A)
            print("b=",b)
            print("x=",x)
        if debug >=1:
            print("estimation,variance:",estimation,variance)

        ret[i,0] = estimation
        ret[i,1] = variance

    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

'''Calculate block kriging for 3D points'''
def kriging3d_block_multipasses(kriging_type,grid,points,tag_in,data,tag_out,vm,discritization=None,mean=None,search_angles=None,search_anisotropy=None,search_passes=None,search_soft=None,full=False,debug=1):
    if mean is None:
        mean = np.mean(data)

    if discritization is not None:
        dblock = grid.discretize(discritization)
        npd = len(dblock)
        max_covariance = 0
        for dp in dblock:
            max_covariance += np.sum(vm.covariance(dp,dblock))

        max_covariance -= vm.nugget*npd
        max_covariance /= npd**2
        print("block covariance",max_covariance)
    else:
        dblock = None
        npd = 1
        max_covariance = vm.max_covariance()
        print("covariance",max_covariance)

    tag_values = list(set(tag_in))
    tag_values.sort()

    n = len(grid)
    ret = np.empty((n,3))
    ret_pass = np.zeros(n,dtype=np.int8)

    for tag in tag_values:
        #create kd3
        kdtree = []
        for search_params in search_passes[tag]:
            kdtree += [KDTree3D(points,search_params.azimuth,search_params.dip,search_params.plunge,search_params.anisotropy)]

        kdtree_soft = KDTree3D(points,search_soft.azimuth,search_soft.dip,search_soft.plunge,search_soft.anisotropy)

        non_estimated = 0

        if full:
            ret_indices = []

        for i,point in enumerate(grid):
            if tag_in[i] == tag:
                '''
                Search is performed by multiple passes
                '''
                data_found = False
                for j,search_params in enumerate(search_passes):
                    d,indices = kdtree[j].search(point,search_params.maxdata,search_params.range)
                    #filter by this tag
                    #include any softdata
                    for soft_tag in search_params.softdata.keys():
                        d_soft,indices_soft = kdtree_soft.search(point,soft_params.maxdata_soft,soft_params.range)


                    if len(d) >= search_params.mindata:
                        data_found = True
                        ret_pass[i] = j+1
                        break

                    if full:
                        ret_indices += [indices]

                    n_data = len(d) + len(d_soft)

                    if debug >=1:
                        print('est at:',point,'neigborhs len:',len(indices))
                        print('est at:',point,'soft len:',len(indices_soft))

                    locations_data = np.empty((n_data,3))
                    locations_data[:len(d),:] = points[indices,:]
                    locations_data[len(d):,:] = points[indices_soft,:]
                    value_data = np.empty(n_data)
                    value_data[:len(d),:] = data[indices]
                    value_data[len(d):,:] = data[indices_soft]

                    estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,locations_data,value_data,dblock)
                else:
                    estimation = np.nan
                    variance = np.nan
                    non_estimated +=1
                if debug >=3:
                    print("ponits:",points[indices])
                    print("A=",A)
                    print("b=",b)
                    print("x=",x)
                if debug >=1:
                    print("estimation,variance:",estimation,variance)

                ret[i,0] = estimation
                ret[i,1] = variance

    if full:
        return ret,non_estimated,ret_indices
    else:
        return ret,non_estimated

'''Calculate block kriging for 3D points'''
def kriging3d_block_iterator(kriging_type,grid,points,data,vm,discretization=None,mean=None,mindata=1,maxdata=1,azimuth=0.0,dip=0.0,plunge=0.0,search_range=np.inf,anisotropy=[1.0,1.0],full=False,debug=1):
    if mean is None:
        mean = np.mean(data)
    #create kd3
    kdtree = KDTree3D(points,azimuth,dip,plunge,anisotropy)

    if discretization is not None:
        dblock = grid.discretize(discretization)
        npd = len(dblock)
        max_covariance = 0
        for dp in dblock:
            max_covariance += np.sum(vm.covariance(dp,dblock))

        max_covariance -= vm.nugget*npd

        max_covariance /= npd**2
    else:
        dblock = None
        npd = 1
        max_covariance = vm.max_covariance()

    print("block covariance",max_covariance)

    for i,point in enumerate(grid):
        d,indices = kdtree.search(point,maxdata,search_range)

        if len(indices) < mindata:
            estimation = np.nan
            variance = np.nan
        else:
            estimation,variance,A,b,x = kriging_system(kriging_type,vm,max_covariance,point,points[indices,:],data[indices],dblock)


        if full:
            yield point,estimation,variance,indices
        else:
            yield point,estimation,variance
