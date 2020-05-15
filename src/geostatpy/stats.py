import numpy as np

def cummulative_distribution(value):
    o = np.sort(value)
    c = np.cumsum(o)
    return c

def get_regular_quantiles(values,bins):
    step = 100.0/bins
    return np.percentile(values,np.arange(0.0,100.0,step)+step)


def probabilities(values,bins):
    n = len(bins)
    h = np.zeros(n+1)
    #print('bins',bins)
    index = np.where(values<=bins[0])[0]
    h[0] = len(index)

    for i in range(n-1):
        b0=bins[i]
        b1=bins[i+1]
        index = np.where(np.logical_and(values>b0,values<=b1))[0]
        h[i+1] = len(index)

    index = np.where(values>bins[-1])[0]
    h[-1] = len(index)

    return h/len(values)

def mask_bins(values,bins):
    '''
    '''
    # First: find the right class
    n = len(values)
    clusters = np.searchsorted(bins,values)

    diff_clusters = set(list(clusters))
    n_c = len(diff_clusters)
    
    ret = np.empty(n,dtype=np.int32)
    for i,c in enumerate(diff_clusters):
        idx = np.where(clusters==c)[0]
        ret[idx] = i

    return ret,n_c

def entropy(probability,verbose=0):
    if (verbose>0): print('probability',*probability)
    s = 0.0
    for p in probability:
        if p > 0:
            s += p*np.log(p)
            if (verbose>0): print('p',p,'log(p)',np.log(p))
    return -s
'''Perform the affine correction of a distribution v
'''
def affine_correction(v,d2,variance_value,mean_value):
    f = 1.0 - d2/variance_value
    #print('f',f)
    #return np.sqrt(f)*(v-mean_value)+mean_value
    return np.sqrt(f)*(v-mean_value)+mean_value

def mean_value_cdf(thresholds,cdf,min_value,max_value,d=200):
    #make a linear interpolator
    dsize = 1.0/d
    n = len(thresholds)
    xp = np.empty(n+2)
    yp = np.empty(n+2)
    if len(cdf) == len(thresholds) + 1:
        xp[1:] = cdf
    else:
        xp[1:-1] = cdf
    xp[0] = 0.0
    xp[-1] = 1.0


    yp[1:-1] = thresholds
    yp[0] = min_value
    yp[-1] = max_value

    tmp = np.interp(np.linspace(dsize*0.5,1,d),xp,yp)
    #print('tmp',tmp)
    mean_value = np.sum(tmp)*dsize
    dcdf = np.interp(tmp,yp,xp)
    #print('dcdf',dcdf)

    #transform cdf into pdf
    pdf = dcdf[1:] - dcdf[:-1]
    #print(pdf,np.sum(pdf)*dsize)

    etype = np.sum(dcdf) / d
    etype2 = np.interp(0.5,xp,yp)

    #print('mean_value',mean_value)
    #print('etype',etype)
    #print('etype2',etype2)

    return mean_value,etype2

def mean_value_cdf2(thresholds,cdf,min_value,max_value,d=500):
    #make a linear interpolator
    dsize = 1.0/d
    n = len(thresholds)
    xp = np.empty(n+2)
    yp = np.empty(n+2)
    if len(cdf) == len(thresholds) + 1:
        xp[1:] = cdf
    else:
        xp[1:-1] = cdf
    xp[0] = 0.0
    xp[-1] = 1.0
    yp[1:-1] = thresholds
    yp[0] = min_value
    yp[-1] = max_value
    dcdf = np.interp(np.linspace(0,1,d)+ dsize*0.5,xp,yp)

    etype = np.sum(dcdf) / d

    etype2 = np.interp(0.5,xp,yp)

    return etype,etype2
if __name__ == "__main__"    :
    data = np.arange(5000)*10

    print("quantile",get_regular_quantiles(data,10))
