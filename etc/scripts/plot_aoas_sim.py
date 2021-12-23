import numpy as np
import matplotlib.pyplot as plt
import pandas

# data = pandas.read_csv('/beaver/backup/icebear/icebear/tools/f1deg.csv', usecols=['x', 'y', 'cx', 'cy'])

# x = data['x']
# y = data['y']
# cx = data['cx']
# cy = data['cy']
#
# error_x = np.abs(x - cx)
# error_y = np.abs(y - cy)
# error = np.sqrt((x - cx)**2 + (y - cy)**2)
#
# plt.figure()
# # plt.hist(y[error>1])
# plt.scatter(x, y, c=cy)
# plt.colorbar()
#
# error_x = error_x.values.reshape((61, 20)).T
# error_y = error_y.values.reshape((61, 20)).T
# error = error.values.reshape((61, 20)).T
#
# plt.figure()
# plt.subplot(311)
# plt.imshow(error_x, origin='lower', interpolation='gaussian', extent=[-30, 30, 0, 20], vmin=1.0, vmax=5.0)
# plt.colorbar()
# plt.subplot(312)
# plt.imshow(error_y, origin='lower', interpolation='gaussian', extent=[-30, 30, 0, 20], vmin=1.0, vmax=5.0)
# plt.colorbar()
# plt.subplot(313)
# plt.imshow(error, origin='lower', interpolation='gaussian', extent=[-30, 30, 0, 20], vmin=1.0, vmax=5.0)
# plt.colorbar()
#
# plt.show()

data1 = pandas.read_csv('/beaver/backup/icebear/icebear/tools/azimuth_extent_acc.csv', usecols=['x', 'y', 'sx', 'sy', 'mx', 'my'])
x1 = data1['x']
y1 = data1['y']
sx1 = data1['sx']
sy1 = data1['sy']
mx1 = data1['mx']
my1 = data1['my']

error_x1 = np.abs(x1 - mx1)
error_y1 = np.abs(y1 - my1)
error1 = np.sqrt((x1 - mx1)**2 + (y1 - my1)**2)

plt.figure()
plt.title('Angle of Arrival Error vs. Target Azimuth Extent')
plt.xlabel('Azimuth Extent [deg]')
plt.ylabel('Angle of Arrival Error [deg]')
plt.plot(sx1, error_x1, 'r', label='azimuth error')
plt.plot(sx1, error_y1, 'b', label='elevation error')
plt.plot(sx1, error1, 'k', label='total error')
plt.legend(loc='upper left')
plt.xlim(0, 50)
plt.ylim(0, 3.0)

data2 = pandas.read_csv('/beaver/backup/icebear/icebear/tools/elevation_extent_acc.csv', usecols=['x', 'y', 'sx', 'sy', 'mx', 'my'])
x2 = data2['x']
y2 = data2['y']
sx2 = data2['sx']
sy2 = data2['sy']
mx2 = data2['mx']
my2 = data2['my']

error_x2 = np.abs(x2 - mx2)
error_y2 = np.abs(y2 - my2 - 0.4)
error2 = np.sqrt((x2 - mx2)**2 + (y2 - my2 - 0.4)**2)

plt.figure()
plt.title('Angle of Arrival Error vs. Target Elevation Extent')
plt.xlabel('Elevation Extent [deg]')
plt.ylabel('Angle of Arrival Error [deg]')
plt.plot(sy2, error_x2, 'r', label='azimuth error')
plt.plot(sy2, error_y2, 'b', label='elevation error')
plt.plot(sy2, error2, 'k', label='total error')
plt.legend(loc='upper left')
# plt.xlim(0, 5.0)
# plt.ylim(0, 3.0)

plt.show()
