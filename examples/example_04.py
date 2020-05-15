import numpy as np
from geostatpy import variogram
from geostatpy import kriging
from geostatpy import geometry

'''This example shows how to do a kriging for a grid using the iterator function
'''

#this is the dummy variogram model. It is not a real one
vmodel = variogram.VariogramModel3D(0.5)
vmodel.add_structure("spherical",1.5,[10,10,10],[0,0,0])
vmodel.add_structure("spherical",1.0,[300,300,300],[0,0,0])

data = np.loadtxt("../data/samples.csv",delimiter=";")

points = data[:,0:3]
ore = data[:,3]

grid = geometry.Grid3D([10,10,10],[40,60,13],[20.0,30.0,6.5])

iterator = kriging.kriging3d_block_iterator("ordinary",grid,points,ore,vmodel,discretization=None,mindata=1,maxdata=5,azimuth=0.0,dip=0.0,plunge=0.0,search_range=100,anisotropy=[1.0,1.0],full=True)

for point,estimation,variance,indices in iterator:
    print("Estimation at {0}: estimated value = {1}, variance = {2}, samples used = {3}".format(point,estimation,variance,len(indices)))

