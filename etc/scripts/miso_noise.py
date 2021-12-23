import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1000)
s1 = np.random.rand(1000) * 1.0
s2 = np.random.rand(1000) * 1.0

s = s1 + s2
n = s1 - s2

fs = np.fft.fftshift(np.fft.fft(s))
fn = np.fft.fftshift(np.fft.fft(n))

plt.figure()
plt.plot(t, s1, 'b')
plt.plot(t, s2, 'r')

plt.figure()
plt.plot(t, fs, '--b')
plt.plot(t, fn, '--r')

plt.show()
