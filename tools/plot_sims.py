import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('poop.h5', 'r')
x1 = f['x1'][()]
y1 = f['y1'][()]
t1 = f['t1'][()]
x2 = f['x2'][()]
y2 = f['y2'][()]
t2 = f['t2'][()]

el = np.arange(0, 90+1, 1)

plt.figure()
plt.title('Cartesian vs. Spherical Low-Elevation Accuracy')
plt.plot(el, el, '--k', label='Actual')
plt.plot(el, y1, 'b', label='Spherical')
plt.plot(el, y2[::-1], 'r', label='Cartesian')
plt.legend(loc='best')
plt.xlabel('Elevation [deg]')
plt.ylabel('Measured Elevation [deg]')
plt.xlim((0, 20))
plt.ylim((0, 20))

plt.figure()
plt.title('Absolute Angle of Arrival Error')
plt.plot(el, np.sqrt((el - y1)**2 + x1**2), 'b', label='Spherical')
plt.plot(el, np.sqrt((el - y2[::-1])**2 + x2[::-1]**2), 'r', label='Cartesian')
plt.legend(loc='best')
plt.xlabel('Elevation [deg]')
plt.ylabel('Error [deg]')
plt.xlim((0, 20))
plt.ylim((0, 5))

plt.figure()
plt.plot(t1, label='Spherical')
plt.plot(t2, label='Cartesian')
plt.legend(loc='best')

plt.show()
