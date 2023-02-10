"""
What I am trying to do is take raw images and combine them into a surface plot of some kind to bascially have
a 3D model of what ever I have imaged.
"""
import pickle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import pymap3d as pm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.path as mpath
import numpy as np
from geographiclib.geodesic import Geodesic

# Shape [rng-dop bins, targets per bin, vertices, degrees of freedom]
file = open('/home/arl203/icebear/icebear/processing/important4', 'rb')
data = pickle.load(file)
fig = plt.figure(figsize=[12, 12])
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
xx = []
yy = []
zz = []
for bin in data:
    for target in bin:
        if target[3, 0] < 3.0:
            continue
        if -1500.0 < target[4, 0] < 1500.0:
        # if 480.0 < target[4, 0] < 520.0:
            x, y, z = pm.geodetic2ecef(target[0, :], target[1, :], target[2, :]*1000, ell=pm.Ellipsoid("wgs84"), deg=True)
            x /= 1000
            y /= 1000
            z /= 1000

            vertices = [list(zip(x, y, z))]
            # Vertices shape [object1[vertex1(x, y, z), vertex2(x, y, z)], object2[(x, y, z)]]
            p = Poly3DCollection(vertices, alpha=0.4, cmap=cm.jet)
            p.set_array(target[4, :])
            p.set_clim([-1500, 1500])
            ax.add_collection3d(p)

            xx += x.tolist()
            yy += y.tolist()
            zz += z.tolist()
        else:
            continue

# import pyvista as pv

xx = np.array(xx)
yy = np.array(yy)
zz = np.array(zz)
verts = np.array([xx, yy, zz]).T

# point_cloud = pv.PolyData(verts)
# point_cloud.plot()
# mesh = point_cloud.reconstruct_surface()
# mesh.plot()

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(verts)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 12.0)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)
# exit()

# Plot the Earth
# r = 6378.0
# u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
# x = r * np.cos(u) * np.sin(v)
# y = r * np.sin(u) * np.sin(v)
# z = r * np.cos(v)
# ax.plot_surface(x, y, z, color='blue', alpha=0.3)


ax.set_xlim(-1500, -500)
ax.set_ylim(-4000, -3000)
ax.set_zlim(5000, 6000)
plt.colorbar(p, shrink=0.5)
plt.show()

