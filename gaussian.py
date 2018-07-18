import numpy as np
from scipy.stats import norm

def made_monotonic(fn):
    for k in range(1,len(fn)):
        if fn[k] <= fn[k-1]:
            if abs(fn[k-1])>1e-14:
                fn[k]=fn[k-1]+1.0e-14
            elif fn[k-1]==0:
                fn[k]= 1e-80
            else:
                fn[k]=fn[k-1]+10**(np.log10(abs(fn[k-1])))


def marginal_uniformization(x,porc=1.0,precision=1000):
    p_aux = (porc/100)*abs(max(x)-min(x))
    R_aux = np.linspace(min(x),max(x),np.sqrt(len(x))+1)

    R = (R_aux[:-1] + R_aux[1:]) / 2.0
    
    R = np.concatenate((R, [max(x)+p_aux]))
    
    C,R = np.histogram(x,bins=int(np.sqrt(len(x))))
    R = (R[:-1] + R[1:]) / 2.0

    #print C,R


    C = np.cumsum(C)
   
    n = np.max(C)

    #print C,R,n

    
    C = (1.0-1.0/n)*C/n

    
    #print C,R
    
    T = {}

    incr_R = (R[1]-R[0])/2.0

    #append 4 values at extremmes
    RN = np.empty(len(R)+3)
    RN[0] = min(x)-p_aux
    RN[1] = min(x)
    RN[2:-1] = R + incr_R
    RN[-1] = max(x)+p_aux+incr_R

    CN = np.empty(len(C)+3)

    CN[0] = 0
    CN[1] = 1.0/n
    CN[2:-1] = C
    CN[-1] = 1.0 

    #print "RN",RN
    #print "CN",CN
    
    Range_2 = np.linspace(RN[0],RN[-1],precision)


    C2 = np.interp(Range_2,RN,CN)
    #print "C2 A",C2.shape,np.min(C2),np.max(C2)
    made_monotonic(C2)
    #print "C2 B",C2.shape,np.min(C2),np.max(C2)
    C2 = C2/max(C2)
    #print "C2 C",C2.shape,np.min(C2),np.max(C2)

    #print "Range_2",Range_2.shape,np.min(Range_2),np.max(Range_2)
    #print "RN",RN.shape,np.min(RN),np.max(RN)
    #print "CN",CN.shape,np.min(CN),np.max(CN)
    #print "C2",C2.shape,np.min(C2),np.max(C2)
    #print "x",x.shape,np.min(x),np.max(x)

    x_lin = np.interp(x,Range_2,C2)
    
    #print "x_lin",np.min(x_lin),np.max(x_lin)

    T["C"] = CN
    T["R"] = RN
    
    return (x_lin,T)
    

def inv_marginal_uniformization(x_lin,T,precision=1000):
    C = T["C"]
    R = T["R"]

    Range2 = np.linspace(R[0],R[-1],precision)
    C2 = np.interp(Range2,R,C)
    made_monotonic(C2)
    C2 = C2/np.max(C2)
    
    x2 = np.interp(x_lin,C2,Range2)
    
    return x2
    

def marginal_gaussianization(x,porc=10.0,precision=1000):
    x_unif, T = marginal_uniformization(x,porc,precision)

    #print "x_unif",np.min(x_unif),np.max(x_unif)

    ret = norm.ppf(x_unif)

    #print "norm.ppf",np.min(ret),np.max(ret)

    return (ret,T)

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
    lower_tails = np.where(data < table[0,1])[0]
    if len(lower_tails) > 0:
        print("lower_tails",table[0,1],lower_tails)
        pass
    
    upper_tails = np.where(data > table[-1,1])[0]
    if len(upper_tails) > 0:
        print("upper_tails",table[-1,1],upper_tails)
        pass
    
    if len(lower_tails) + len(upper_tails) > 0:
        #there are tails
        #cond = (data >= table[0,0])
        #cond = cond and data >= table[0,0]
        pass
        #tmp = np.where(cond)[0]
    
    backtransformed_data = np.interp(data, table[:,1], table[:,0])

    return backtransformed_data

class NormalScoreTransformer(object):
    def __init__(self):
        self._transformed_data = None
        self._table = None
        self._dim = None
        
    def fit(self,X,missing_value=-999):
        self._dim = len(X.shape)
        
        if len(X.shape) > 1:
            self._dim = X.shape[1]
            self._transformed_data = np.empty_like(X)
            self._table =  []
            for i in range(self._dim):
                #find missings
                idx = np.where(X[:,i] != missing_value)[0]
                
                #print "missings",missing_value,i,len(idx)
                
                td, t = normal_scores(X[:,i],indices=idx,na_value=missing_value)
                
                self._transformed_data[:,i] = td
                self._table +=  [t]
        else:
            self._dim = 1
            self._transformed_data, self._table =  normal_scores(X)
             
    def fit_transform(self,X,missing_value=-999):
        self.fit(X,missing_value=missing_value)
        return self._transformed_data
        
    def inverse_transform(self,Y):
        if self._dim > 1:
            backtransformed_data = np.empty_like(Y)
            for i in range(self._dim):
                #find missings
                #idx = np.where(X[:,i] != missing_value)[0]
                
                back_values = back_normal_scores(Y[:,i],self._table[i])

                backtransformed_data[:,i] = back_values
            return backtransformed_data
        else:
            return back_normal_scores(Y,self._table)
    
if __name__ == "__main__"    :
    data = np.arange(5000)*10
    transformed_data, table =  normal_scores(data)    
    bdata = back_normal_scores(transformed_data,table)

    print("mean",np.mean(transformed_data))
    print("std",np.std(transformed_data))
    print("RSME",np.mean((data-bdata)**2))
