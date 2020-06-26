import numpy as np

import sys

sys.path += ['../src']

from geostatpy import search
from geostatpy import geometry
from geostatpy import id

'''This example shows how to make estimations with the inverse of the distance approach
'''

#load the data
data = np.loadtxt("../data/samples.csv",delimiter=";")

#make locations and variable to work with
points = data[:,0:3]
ore = data[:,3]

#define a grid to estimate
grid = geometry.Grid3D([10,10,10],[40,60,13],[20.0,30.0,6.5])

#create kd3 for searching
kdtree = search.KDTree3D(points=points,azimuth=0.0,dip=0.0,plunge=0.0,anisotropy=[1.0,1.0])

#estimate by inverse of distance
estimations = id.inverse_distance(grid,ore,mindata=1,maxdata=24,search_range=50.0,kdtree=kdtree,p=1.5)

for i,est in enumerate(estimations):
    print("Estimation at {0} = {1}".format(grid[i],est))

