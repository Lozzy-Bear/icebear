import matplotlib.pyplot as plt
import numpy as np

# Numpy

fig, ax = plt.subplots(6, 1)

noise_mag = 2

# Sampling
samp_rate = 1000  # samples per second
symbol_rate = 10  # symbols (bits) per second
sps = samp_rate / symbol_rate  # samples per symbol
t0 = 0.0  # start time
tf = 10.0  # end time
samples = np.arange(samp_rate * tf)  # integer linspace (indices)
t = np.linspace(t0, tf, samples.size)  # timespace (seconds)


# Carrier wave
freq = 5.0  # carrier frequency (hertz)
amp = 1.0  # carrier amplitude
carrier = amp * np.sin(2 * np.pi * freq * t)  # carrier wave

ax[0].plot(t, carrier)
ax[0].set_title('Carrier')

# Bit Sequence
bits_unexpanded = np.random.randint(2, size=(int(symbol_rate * (tf-t0))))  # generates binary array, not same size as carrier wave yet
bits = np.repeat(bits_unexpanded, sps)  # expands the bit sequence to match samples per symbol

ax[1].plot(t, bits)
ax[1].set_title('Data')

# BPSK mod
modulated = np.zeros(t.size)  # phase modulation
modulated[bits == 1] = amp * np.sin(2 * np.pi * freq * t[bits == 1])
modulated[bits == 0] = amp * np.sin(2 * np.pi * freq * t[bits == 0] + np.pi)

ax[2].plot(t, modulated)
ax[2].set_title('Modulated')

# add noise
noise = np.random.normal(0, noise_mag*amp, t.size)
noisy_modulated = modulated + noise

ax[3].plot(t, noisy_modulated)
ax[3].set_title('Added Noise')


# BPSK demod (assume we recover the carrier phase - normally done using PLL or Costas loop)

# multiply received signal by reference frequency signal (normally comes from PLL)
noisy_multiplied = noisy_modulated * np.sin(2 * np.pi * freq * t)

ax[4].plot(t, noisy_multiplied)
ax[4].set_title('Received Signal after Multiplication by Ref Freqency')


demodulated = np.zeros(t.size)
# integrate over symbol period
for i in range(0, t.size-1, int(sps)):
    demodulated[i:i+int(sps)] = np.trapz(noisy_multiplied[i:i+int(sps)], x=t[i:i+int(sps)])

# threshold selector
demodulated[demodulated > 0] = 1
demodulated[demodulated < 0] = 0


ax[5].plot(t, demodulated)
ax[5].set_title('Demodulated Data')

plt.show()

