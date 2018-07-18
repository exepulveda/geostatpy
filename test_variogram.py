import numpy as np
import variogram
import kriging

vmodel = variogram.VariogramModel3D(0.5)
vmodel.add_structure("spherical",1.5,[10,10,10],[0,0,0])
vmodel.add_structure("spherical",1.0,[300,300,300],[0,0,0])


data = np.loadtxt("muestras.csv",delimiter=";")
points = data[:,0:3]
cut = data[:,3]

#create random points in 100,100,100
#n = 2000
#points = np.random.random((n,3))
#points *= 200.0

#print points[0],points[1],np.sum((points[0]-points[1])**2)

#for i in xrange(n):
#    for j in xrange(i,n):
#        ret = vmodel.covariance(points[i],points[j])
#        print i,j,ret


directions = [(150,1.0,0,0),(100,2.5,90,0)]

#ret = vmodel.compute_variogram(directions)
#for vmodel,cmax in ret:
#    print vmodel,cmax

#A,b = vmodel.kriging_system(points[0,:],points[20:24,:])
#print A,b

ret,err,non_estimated,errors,ret_indices = kriging.kriging3d_cross_validation("ordinary",points,cut,vmodel,mindata=1,maxdata=5,azimuth=0.0,dip=0.0,plunge=0.0,search_range=100,anisotropy=[1.0,1.0],full=True)

np.savetxt("kg_out.csv",ret)
np.savetxt("kg_out_indices",ret_indices)

print err

