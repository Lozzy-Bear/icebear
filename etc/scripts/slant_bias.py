import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = [0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9]  # [x,...]
y = [0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.]  # [y,...]
z = [0., 0.0895, 0.3474, 0.2181, 0.6834, -0.0587, -1.0668, -0.7540, -0.5266, -0.4087]  # [z,...]

# x = np.asarray(x)
# y = np.asarray(y)
# z = np.asarray(z)
#
# sx = 90-np.rad2deg(np.arctan2(x, z))
# sy = np.rad2deg(np.arctan2(y, z))
#
# print(np.average(sx), sx)
# print(np.average(sy), sy)

# These constants are to create random data for the sake of this example
N_POINTS = 10
TARGET_X_SLOPE = 2
TARGET_y_SLOPE = 3
TARGET_OFFSET  = 5
EXTENTS = 5
NOISE = 5

# Create random data.
# In your solution, you would provide your own xs, ys, and zs data.
xs = x
ys = y
zs = z

# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')

# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

# Manual solution
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
print("errors: \n", errors)
print("residual:", residual)

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

