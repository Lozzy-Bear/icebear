import numpy as np
from scipy.integrate import dblquad
import multiprocessing as mp
import h5py


class Target:
    def __init__(self, targets,
                 antennas=np.array([[0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9],
                                    [0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.],
                                    [0., 0.0895, 0.3474, 0.2181, 0.6834, -0.0587, -1.0668, -0.7540, -0.5266, -0.4087]]),
                 frequency=49.5e6, doppler=0.0):
        print(f'\ttarget: {targets}')
        self.wavelength = 299792458 / (frequency + doppler)
        self.x = antennas[0, :]
        self.y = antennas[1, :]
        self.z = antennas[2, :]
        self.u = np.array([0])
        self.v = np.array([0])
        self.w = np.array([0])
        for i in range(len(self.x)):
            for j in range(i + 1, len(self.x)):
                self.u = np.append(self.u, (self.x[i] - self.x[j]) / self.wavelength)
                self.v = np.append(self.v, (self.y[i] - self.y[j]) / self.wavelength)
                self.w = np.append(self.w, (self.z[i] - self.z[j]) / self.wavelength)
        self.u = np.append(self.u, -1 * self.u)
        self.v = np.append(self.v, -1 * self.v)
        self.w = np.append(self.w, -1 * self.w)
        self.target_az = np.deg2rad(targets[:, 0])
        self.target_ax = np.deg2rad(targets[:, 1])
        self.target_el = np.deg2rad(targets[:, 2])
        self.target_ex = np.deg2rad(targets[:, 3])

        idx_length = len(targets)
        num_baselines = len(self.u)
        self.visibility = np.zeros(num_baselines, dtype=np.complex64)
        for idx in range(idx_length):
            for ant in range(num_baselines):
                self._visibility_calculation(ant, self.u[ant], self.v[ant], self.w[ant],
                                             self.target_az[idx], self.target_ax[idx],
                                             self.target_el[idx], self.target_ex[idx])

    def _real_pre_integrate(self, theta, phi, u_in, v_in, w_in, theta_mean, theta_spread, phi_mean, phi_spread):
        return 2 * np.real(np.exp(-(theta - theta_mean) ** 2 / (2.0 * theta_spread * theta_spread)) *
                           np.exp(-(phi - phi_mean) ** 2 / (2.0 * phi_spread * phi_spread)) * np.cos(phi) *
                           np.exp(-2.0j * np.pi * ((u_in * np.sin(theta) * np.cos(phi)) +
                                                    (v_in * np.cos(theta) * np.cos(phi)) + (w_in * np.sin(phi)))))

    def _imag_pre_integrate(self, theta, phi, u_in, v_in, w_in, theta_mean, theta_spread, phi_mean, phi_spread):
        return 2 * np.imag(np.exp(-(theta - theta_mean) ** 2 / (2.0 * theta_spread * theta_spread)) *
                           np.exp(-(phi - phi_mean) ** 2 / (2.0 * phi_spread * phi_spread)) * np.cos(phi) *
                           np.exp(-2.0j * np.pi * ((u_in * np.sin(theta) * np.cos(phi)) +
                                                    (v_in * np.cos(theta) * np.cos(phi)) + (w_in * np.sin(phi)))))

    def _visibility_calculation(self, idx, u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread):
        real_vis = dblquad(self._real_pre_integrate, -np.pi / 2, np.pi / 2, lambda phi: -np.pi, lambda phi: np.pi,
                           args=(u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread))[0]
        imag_vis = dblquad(self._imag_pre_integrate, -np.pi / 2, np.pi / 2, lambda phi: -np.pi, lambda phi: np.pi,
                           args=(u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread))[0]
        self.visibility[idx] += real_vis - imag_vis * 1.0j  # the - 1j just rotates it to match previous sims


def generate_batch_sim(az_min, az_max, el_min, el_max, az_ext=3.0, el_ext=3.0, resolution=1.0):
    az = np.arange(az_min, az_max+resolution, resolution)
    el = np.arange(el_min, el_max+resolution, resolution)
    targets = []
    for a in az:
        for e in el:
            targets.append(np.array([[a, az_ext, e, el_ext], ]))

    return targets


def get_target_visibility(x):
    t = Target(x)
    return t.visibility


if __name__ == '__main__':
    targets = generate_batch_sim(-10.0, 10.0, 5.0, 10.0)
    print(f'starting multiprocessing (cores={int(mp.cpu_count()/2)}) simulating:')
    pool = mp.Pool(processes=int(mp.cpu_count()/2))
    visibilities = pool.map(get_target_visibility, targets)
    filename = 'simulated_data.h5'
    f = h5py.File(filename, 'w')
    f.create_dataset('targets', data=targets)
    f.create_dataset('visibilities', data=visibilities)
    f.close()
    print(f'successful completion: file={filename}')





