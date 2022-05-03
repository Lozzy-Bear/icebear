import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
# Cupy

fig, ax = plt.subplots(6, 1)

noise_mag = 2

# Sampling
samp_rate = 1000  # samples per second
symbol_rate = 10  # symbols (bits) per second
sps = samp_rate / symbol_rate  # samples per symbol
t0 = 0.0  # start time
tf = 10.0  # end time
samples = cp.arange(samp_rate * tf)  # integer linspace (indices)
t = cp.linspace(t0, tf, samples.size)  # timespace (seconds)


# Carrier wave
freq = 5.0  # carrier frequency (hertz)
amp = 1.0  # carrier amplitude
carrier = amp * cp.sin(2 * np.pi * freq * t)  # carrier wave

ax[0].plot(cp.asnumpy(t), cp.asnumpy(carrier))
ax[0].set_title('Carrier')

# Bit Sequence
bits_unexpanded = cp.random.randint(2, size=(int(symbol_rate * (tf-t0))))  # generates binary array, not same size as carrier wave yet
bits = cp.repeat(bits_unexpanded, int(sps))  # expands the bit sequence to match samples per symbol

ax[1].plot(cp.asnumpy(t), cp.asnumpy(bits))
ax[1].set_title('Data')

# BPSK mod
modulated = cp.zeros(t.size)  # phase modulation
modulated[bits == 1] = amp * cp.sin(2 * np.pi * freq * t[bits == 1])
modulated[bits == 0] = amp * cp.sin(2 * np.pi * freq * t[bits == 0] + np.pi)

ax[2].plot(cp.asnumpy(t), cp.asnumpy(modulated))
ax[2].set_title('Modulated')

# add noise
noise = cp.random.normal(0, noise_mag*amp, t.size)
noisy_modulated = modulated + noise

ax[3].plot(cp.asnumpy(t), cp.asnumpy(noisy_modulated))
ax[3].set_title('Added Noise')


# BPSK demod (assume we recover the carrier phase - normally done using PLL or Costas loop)

# multiply received signal by reference frequency signal (normally comes from PLL)
noisy_multiplied = noisy_modulated * cp.sin(2 * np.pi * freq * t)

ax[4].plot(cp.asnumpy(t), cp.asnumpy(noisy_multiplied))
ax[4].set_title('Received Signal after Multiplication by Ref Freqency')


demodulated = cp.zeros(t.size)
# integrate over symbol period
for i in range(0, t.size-1, int(sps)):
    demodulated[i:i+int(sps)] = np.trapz(cp.asnumpy(noisy_multiplied[i:i+int(sps)]), x=cp.asnumpy(t[i:i+int(sps)]))

# threshold selector
demodulated[demodulated > 0] = 1
demodulated[demodulated < 0] = 0


ax[5].plot(cp.asnumpy(t), cp.asnumpy(demodulated))
ax[5].set_title('Demodulated Data')

plt.show()
