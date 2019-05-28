import numpy as np
import numpy.ma as ma
import scipy.spatial

def inverse_distance_loc(loc,values,mindata,maxdata,search_range,kdtree,p=1.5,full=False):
    if full:
        ret_indices = []

    #serach one more to eliminate itself
    d,indices = kdtree.search(loc,maxdata+1,search_range)

    #remove zero distance
    if d[0] <= 0.00000001:
        est = values[indices[0]]
        return est,[indices[0]]

    d =  d[1:]
    indices = indices[1:]

    if full:
        ret_indices = indices

    #inverse
    id = 1.0/d**p
    sum_id = np.sum(id)
    weights = id / sum_id

    estimation = np.sum(values[indices] * weights)

    #print(loc,d,indices,weights)

    if full:
        return estimation,ret_indices
    else:
        return estimation

def inverse_distance_weights(loc,mindata,maxdata,search_range,kdtree,p=1.5):
    #serach one more to eliminate itself
    d,indices = kdtree.search(loc,maxdata+1,search_range)

    #remove zero distance
    if d[0] <= 0.00000001:
        return (indices[0],np.array([1.0]))

    d =  d[1:]
    indices = indices[1:]

    #inverse
    id = 1.0/d**p
    sum_id = np.sum(id)
    weights = id / sum_id

    return (indices,weights)

def inverse_distance(points,values,mindata,maxdata,search_range,kdtree,p=1.5,full=False):
    n = len(points)
    ret = np.empty(n)
    mask = np.empty(n,dtype=bool)
    mask.fill(False)

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
            mask[i] = True
        else:
            #inverse
            id = 1.0/d**p
            sum_id = np.sum(id)
            weights = id / sum_id

            estimation = np.sum(values[indices] * weights)

            ret[i] = estimation

    ret = ma.masked_array(ret,mask=mask)
    if full:
        return ret,ret_indices
    else:
        return ret
