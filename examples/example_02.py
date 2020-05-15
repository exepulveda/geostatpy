import numpy as np
from geostatpy import variogram
from geostatpy import kriging

'''This example shows how to do a cross validation by kriging 2D case
'''
vmodel = variogram.VariogramModel2D(0.5)
vmodel.add_structure("spherical",1.5,[10,10],0)
vmodel.add_structure("spherical",1.0,[300,300],0)

#original 3d
data = np.loadtxt("../data/samples.csv",delimiter=";")
points = data[:,0:3]
ore = data[:,3]

#create 2d by keeping only unique x,y (just first z is stored)
keys = set()
points2d = []
for d in points:
    if (d[0],d[1]) not in keys:
        keys.add((d[0],d[1]))
        points2d += [(d[0],d[1])]
    
points2d = np.array(points2d)

ret,err,non_estimated,errors,ret_indices = kriging.kriging2d_cross_validation("ordinary",points2d,ore,vmodel,mindata=1,maxdata=5,azimuth=0.0,search_range=100,anisotropy=1.0,full=True)

for i,point in enumerate(points2d):
    estimation = ret[i,0]
    error = errors[i]
    indices = ret_indices[i]
    print("Crossvalidation at {0}: real value = {1}, estimated value = {2}, error = {3}, samples used = {4}".format(point,ore[i],estimation,error,len(indices)))
