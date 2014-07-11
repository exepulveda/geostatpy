import numpy as np
import geometry
import variogram
import kriging

'''This example shows how to do a kriging for a grid using discretized block of 3x3x2
'''

#this is the dummy variogram model. It is not a real one
vmodel = variogram.VariogramModel3D(0.5)
vmodel.add_structure("spherical",1.5,[10,10,10],[0,0,0])
vmodel.add_structure("spherical",1.0,[300,300,300],[0,0,0])

data = np.loadtxt("samples.csv",delimiter=";")

points = data[:,0:3]
ore = data[:,3]

grid = geometry.Grid3D([10,10,10],[40,60,13],[20.0,30.0,6.5])

ret,non_estimated,ret_indices = kriging.kriging3d_block("ordinay",grid,points,ore,vmodel,[3,3,2],mindata=1,maxdata=15,azimuth=0.0,dip=0.0,plunge=0.0,search_range=100,anisotropy=[1.0,1.0],full=True)

for point,r,indices in zip(grid,ret,ret_indices):
    estimation,variance = r
    print("Estimation at {0}: estimated value = {1}, variance = {2}, samples used = {3}".format(point,estimation,variance,len(indices)))
