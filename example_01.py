import numpy as np
import variogram
import kriging

'''This example shows how to do a cross validation by kriging
'''

#this is the dummy variogram model. It is not a real one
vmodel = variogram.VariogramModel3D(0.5)
vmodel.add_structure("spherical",1.5,[10,10,10],[0,0,0])
vmodel.add_structure("spherical",1.0,[300,300,300],[0,0,0])


data = np.loadtxt("samples.csv",delimiter=";")

points = data[:,0:3]
ore = data[:,3]

ret,err,non_estimated,errors,ret_indices = kriging.kriging3d_cross_validation("ordinay",points,ore,vmodel,mindata=1,maxdata=5,azimuth=0.0,dip=0.0,plunge=0.0,search_range=100,anisotropy=[1.0,1.0],full=True)

for i,point in enumerate(points):
    estimation = ret[i,0]
    error = errors[i]
    indices = ret_indices[i]
    print "Crossvalidation at {0}: real value = {1}, estimated value = {2}, error = {3}, samples used = {4}".format(point,ore[i],estimation,error,len(indices))
