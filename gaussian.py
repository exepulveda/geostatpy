import numpy as np
from scipy.stats import norm

def normal_scores(o_data,o_weights=None,indices=None,na_value=np.nan):
    epsilon = 1.0e-7
    
    m = len(o_data)

    if o_weights is None:
        o_weights = np.ones(m)
    
    if indices is not None:
        data = o_data[indices]
        weights = o_weights[indices]
    else:
        data = o_data
        weights = o_weights
        
    n = len(data)
    
    data = data + np.random.random(n) * epsilon
    
    #sort data and weights
    table = np.empty((n,2))
    table[:,0] = data
    table[:,1] = weights
    
    table = table[table[:,0].argsort()]
    
    sorted_data = table[:,0]
    sorted_weights = table[:,1]
    
    #normalize weights
    wsum = np.sum(sorted_weights)
    nweights = sorted_weights/wsum #normalized weights
    
    #Cummulative distribution
    cumsum = np.cumsum(nweights) - 0.5/wsum #centroids
    weights = norm.ppf(cumsum)

    #transformation table
    table[:,0] = sorted_data
    table[:,1] = weights
    
    #Interpolate
    transformed_data = np.interp(data, sorted_data, weights)
    

    if indices is not None:
        tmp = np.empty(m)
        tmp[:] = na_value
        tmp[indices] = transformed_data
        transformed_data = tmp

    return transformed_data, table

def back_normal_scores(o_data,table,indices=None,na_value=np.nan):
    epsilon = 1.0e-7
    
    m = len(o_data)

    if indices is not None:
        data = o_data[indices]
    else:
        data = o_data
        
    n = len(data)
    
    #find tails
    lower_tails = np.where(data < table[0,0])[0]
    if len(lower_tails):
        pass
    
    upper_tails = np.where(data > table[-1,0])[0]
    if len(upper_tails):
        pass
    
    if len(lower_tails) + len(upper_tails) > 0:
        #there are tails
        cond = data >= table[0,0]
        cond = cond and data >= table[0,0]
        
        tmp = np.where(cond)[0]
    
    backtransformed_data = np.interp(data, table[:,1], table[:,0])

    return backtransformed_data
    
if __name__ == "__main__"    :
    data = np.arange(5000000)*10

    data[10] = -999
    indices = np.where(data != -999)[0]

    transformed_data, table =  normal_scores(data,indices=indices)

    #import matplotlib.pyplot as plt

    data = np.arange(50)*10
    transformed_data, table =  normal_scores(data)    
    bdata = back_normal_scores(transformed_data,table)

    print data,bdata

    #plt.hist(transformed_data[indices],51)

    #plt.show()
