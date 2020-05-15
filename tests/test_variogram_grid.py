import numpy as np
import variogram
import geometry
import kriging

vmodel = variogram.VariogramModel3D(0.5)
vmodel.add_structure("spherical",1.5,[10,10,10],[0,0,0])
vmodel.add_structure("spherical",1.0,[300,300,300],[0,0,0])


data = np.loadtxt("muestras.csv",delimiter=";")

#create 2d
keys = set()
data2d = []
for d in data:
    if (d[0],d[1]) not in keys:
        keys.add((d[0],d[1]))
        data2d += [d]
    
data2D = np.array(data2d)

points = data[:,0:3]
cut = data[:,3]


grid = geometry.Grid3D([10,10,1],[40,60,130],[20,30,65])

ret,non_estimated,ret_indices = kriging.kriging3d_block("ordinary",grid,points,cut,vmodel,None,mindata=1,maxdata=5,azimuth=0.0,dip=0.0,plunge=0.0,search_range=100,anisotropy=[1.0,1.0],full=True)

print ret

