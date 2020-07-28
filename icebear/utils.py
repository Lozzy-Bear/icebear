import numpy as np
import h5py
import yaml
import datetime
from dateutil.tz import tzutc


def uvw_to_rtp(u, v, w):
    """
    Converts u, v, w cartesian baseline coordinates to radius, theta, phi 
    spherical coordinates.

    Parameters
    ----------
        u : float np.array
            East-West baseline coordinate divided by wavelength.
        v : float np.array
            North-South baseline coordinate divided by wavelength.
        w : float np.array
            Altitude baseline coordinate divided by wavelength.

    Returns
    -------
        r : float np.array
            Radius baseline coordinate divided by wavelength.
        t : float np.array
            Theta (elevation) baseline coordinate.
        p : float np.array
            Phi (azimuthal) baseline coordinate.
    """

    r = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    t = np.pi / 2 - np.arctan2(w, np.sqrt(u ** 2 + v ** 2))
    p = np.arctan2(v, u) + np.pi
    np.nan_to_num(t, copy=False)

    return r, t, p


def rtp_to_uvw(r, t, p):
    """
    Converts radius, theta, phi spherical baseline coordinates to u, v, w
    cartesian coordinates.

    Parameters
    ----------
        r : float np.array
            Radius baseline coordinate divided by wavelength.
        t : float np.array
            Theta (elevation) baseline coordinate.
        p : float np.array
            Phi (azimuthal) baseline coordinate.

    Returns
    -------
        u : float np.array
            East-West baseline coordinate divided by wavelength.
        v : float np.array
            North-South baseline coordinate divided by wavelength.
        w : float np.array
            Altitude baseline coordinate divided by wavelength.
    """

    u = r * np.sin(t) * np.cos(p)
    v = r * np.sin(t) * np.sin(p)
    w = r * np.cos(t)

    return u, v, w


def baselines(x, y, z, wavelength):
    """
    Given relative antenna positions in cartesian coordinates with units of meters
    and the wavelength in meters determines the u, v, w baselines in cartesian coordinates.

    Parameters
    ----------
        x : float np.array
            East-West antenna coordinate in meters.
        y : float np.array
            North-South antenna coordinate in meters.
        z : float np.array
            Altitude antenna coordinate in meters.
        wavelength : float
            Radar signal wavelength in meters.

    Returns
    -------
        u : float np.array
            East-West baseline coordinate divided by wavelength.
        v : float np.array
            North-South baseline coordinate divided by wavelength.
        w : float np.array
            Altitude baseline coordinate divided by wavelength.

    Notes
    -----
        * Given N antenna then M=N(N-1)/2 unique baselines exist.
        * M baselines can include conjugates and a origin baseline for M_total = M*2 + 1.

    Todo
        * Makes options to include or disclude 0th baseline and conjugates.
        * Make array positions load from the calibration.ini file.
        * Error handling for missing antenna position values (like no z).
    """

    # Baseline for an antenna with itself.
    u = np.array([0])
    v = np.array([0])
    w = np.array([0])
    # Include all possible baseline combinations.
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            u = np.append(u, (x[i] - x[j]) / wavelength)
            v = np.append(v, (y[i] - y[j]) / wavelength)
            w = np.append(w, (z[i] - z[j]) / wavelength)
    # Include the conjugate baselines.
    u = np.append(u, -1 * u)
    v = np.append(v, -1 * v)
    w = np.append(w, -1 * w)

    return u, v, w


def walk_hdf5(filepath):
    """
    Walks the tree for a given hdf5 file.

    Parameters
    ----------
        filepath : string
            File path and name to hdf5 file to be walked.

    Returns
    -------
        None
    """

    f = h5py.File(filepath, 'r')
    f.visititems(_print_attrs)
    return None


def walk_yaml(filepath):
    """
    Walks the tree for a given yaml file.

    Parameters
    ----------
        filepath : string
            File path and name to yaml file to be walked.

    Returns
    -------
        None
    """

    f = yaml.full_load(open(filepath))
    print(yaml.dump(f))
    return None


def _print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return None


def fov_window(coeffs, resolution=np.array([0.1, 0.1]),
               azimuth=np.array([225.0, 315.0]), elevation=np.array([90.0, 110.0])):
    """
    Set the field-of-view (fov) for a coefficients set. A narrower fov will result in
    faster run times.

    Parameters
    ----------
        coeffs : complex64 np.array
            Array of pre-calculated SWHT coefficents for full sphere.
        resolution : float np.array
            [Azimuth, Elevation] resolution with minimum 0.1 degrees.
        azimuth : float np.array
            [[start, stop],...] angles within 0 to 360 degrees.
        elevation : float np.array
            [[start, stop],...] angles within 0 to 180 degrees.

    Returns
    -------
        fov_coeffs : complex64 np.array
            Array of pre-calculated SWHt coefficients for FOV.

    Notes
    -----
        * All azimuth field of view zones must have a corresponding elevation zone specified.
        * It is advised to specify field of view zones slightly larger than required.
        * Azimuth and elevation resolution are best kept equal.
        * Boresight is at 270 degrees azimuth and 90 degrees elevation.

    todo
        * function is not complete DO NOT USE.
    """

    fov_coeffs = np.array([])
    for i in range(azimuth.shape[1]):
        az_index = int(azimuth[i, :] / 0.1)
        el_index = int(elevation[i, :] / 0.1)
        az_step = int(resolution[i, 0] / 0.1)
        el_step = int(resolution[i, 1] / 0.1)
        fov_coeffs = np.append(coeffs[el_index[0]:el_index[1]:el_step, \
                               az_index[0]:az_index[1]:az_step, :], axis=0)
    return fov_coeffs


class Time:
    def __init__(self, start, stop, step):
        """
        Class which hold the iteration time series in both human readable and seconds since epoch (1970-01-01) formats.

        Parameters
        ----------
            start : list int
                Start point of time series in format [year, month, day, hour, minute, second, microsecond]
            stop : list int
                Stop point of time series in format [year, month, day, hour, minute, second, microsecond]
            step : list int
                Step size of time series in format [day, hour, minute, second, microsecond]
        """
        if len(start) != 7:
            raise ValueError('Must include [year, month, day, hour, minute, second, microsecond]')
        if len(stop) != 7:
            raise ValueError('Must include [year, month, day, hour, minute, second, microsecond]')
        if len(step) != 5:
            raise ValueError('Must include [day, hour, minute, second, microsecond]')
        self.start_human = datetime.datetime(year=start[0], month=start[1], day=start[2], hour=start[3],
                                             minute=start[4], second=start[5], microsecond=start[6], tzinfo=tzutc())
        self.stop_human = datetime.datetime(year=stop[0], month=stop[1], day=stop[2], hour=stop[3],
                                            minute=stop[4], second=stop[5], microsecond=stop[6], tzinfo=tzutc())
        self.step_human = datetime.timedelta(days=step[0], hours=step[1], minutes=step[2],
                                             seconds=step[3], microseconds=step[4])
        self.start_epoch = self.start_human.timestamp()
        self.stop_epoch = self.stop_human.timestamp()
        self.step_epoch = self.step_human.total_seconds()

    def get_date(self, timestamp):
        return datetime.datetime.fromtimestamp(timestamp, tz=tzutc())
