"""
Dataclass containers for ICEBEAR-3D data processing and HDF5 data packing.
"""
from dataclasses import dataclass, field, fields
import numpy as np
import h5py
from common.utils import version, created


__version__ = version()
__created__ = created()


@dataclass(kw_only=True, slots=True)
class Info:
    date_created: str = field(
        default=__created__,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 19,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the date and time this file was generated'})
    date: np.ndarray((3, ), dtype=int) = field(
        default=np.asarray([0, 0, 0], dtype=int),
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': (3,),
                  'version': __version__,
                  'created': __created__,
                  'description': 'starting date of the data contained within'})
    experiment_name: str = field(
        default='normal',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 6,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of experiment ran (ex; normal, mobile)'})
    radar_config: str = field(
        default='ib3d',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 4,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of the radar the data was collected with (ex; ib, ib3d, lofar)'})
    center_freq: float = field(
        default=49_500_000.0,
        metadata={'type': 'float64',
                  'units': 'hertz',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'radar center frequency in Hz'})
    decimation_rate: int = field(
        default=200.0,
        metadata={'type': 'float64',
                  'units': 'sample/second',
                  'shape': None,
                  'version': __version__,
                   'created': __created__,
                  'description': 'the samples/second to decimate the received signal by'})
    incoherent_averages: float = field(
        default=10,
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the number of incoherent averages to sum'})
    number_ranges: int = field(
        default=2000,
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the number of range gates to process'})
    range_resolution: float = field(
        default=1.5,
        metadata={'type': 'float64',
                  'units': 'kilometer',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the resolution of the range gates in kilometers'})
    clutter_gates: int = field(
        default=100,
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the number of range gates starting from gate 0 that are considered dominated by clutter'})
    timestamp_correction: int = field(
        default=30,
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'offset to correct for the time stamping difference between TX and RX FPGA propagation delays'})
    time_resolution: float = field(
        default_factory=float,
        metadata={'type': 'float64',
                  'units': 'second',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'chip width times code length'})
    coherent_integration_time: float = field(
        default_factory=float,
        metadata={'type': 'float64',
                  'units': 'second',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the coherent integration time is the time resolution times number of incoherent averages'})
    snr_cutoff_db: float = field(
        default=1.0,
        metadata={'type': 'float64',
                  'units': 'decibel',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'signal-to-noise ratio cutoff under which spectra and cross-spectra data is thrown out'})
    spectra_descriptors: np.ndarray((2, 10), dtype=int) = field(
        default=np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=int),
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': (2, 10),
                  'version': __version__,
                  'created': __created__,
                  'description': 'helpful indexing descriptors aid for the ordering of spectra data sets'})
    xspectra_descriptors: np.ndarray((2, 10), dtype=int) = field(
        default=np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]], dtype=int),
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': (2, 45),
                  'version': __version__,
                  'created': __created__,
                  'description': 'helpful indexing descriptors aid for the ordering of xspectra data sets'})
    tx_site_name: str = field(
        default='prelate',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 7,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of the transmitter site'})
    tx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
        default=np.asarray([50.893, -109.403], dtype=float),
        metadata={'type': 'float64',
                  'units': 'degree',
                  'shape': (2,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[latitude, longitude] global North-Easting coordinates of the site in degrees'})
    tx_heading: float = field(
        default=16.0,
        metadata={'type': 'float64',
                  'units': 'degree',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'transmitter array boresight pointing direction in degrees East of North'})
    tx_rf_path: str = field(
        default='X300->amplifier->bulkhead->feedline->antenna',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 44,
                  'version': __version__,
                  'created': __created__,
                  'description': 'RF hardware signal path chain string'})
    tx_ant_type: str = field(
        default='Cushcraft A50-5S',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 16,
                  'version': __version__,
                  'created': __created__,
                  'description': 'brand and model of the antenna used'})
    tx_ant_coords: np.ndarray((3, 10), dtype=float) = field(
        default=np.asarray([[0., 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
        metadata={'type': 'float64',
                  'units': 'meter',
                  'shape': (3, 10),
                  'version': __version__,
                  'created': __created__,
                  'description': '[[x0, ...],[y0, ...],[z0, ...]] transmitter antenna locations in meters from antenna 0'})
    tx_feed_corr: np.ndarray = field(
        default=np.asarray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
        metadata={'type': 'float64',
                  'units': 'volt, degree',
                  'shape': (2, 10),
                  'version': __version__,
                  'created': __created__,
                  'description': '[[ant0 magnitude, ...],[ant0 phase, ...]] feedline calibration correction per antenna'})
    tx_feed_corr_date: np.ndarray((3,), dtype=int) = field(
        default=np.asarray([0, 0, 0]),
        metadata={'type': 'float64',
                  'units': 'year, month, day',
                  'shape': (3,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[year, month, day] date of the last feed line calibration correction'})
    tx_feed_corr_type: str = field(
        default='manual',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'type of procedure used to measure calibration values (ex; manual, closure angle)'})
    tx_ant_mask: np.ndarray((10,), dtype=int) = field(
        default=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': (10,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[ant0, ...] boolean mask (0 off, 1 on) indicating which antennas were used and/or functional'})
    tx_sample_rate: float = field(
        default=800_000.0,
        metadata={'type': 'float64',
                  'units': 'hertz',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'sample rate of transmitted code in Hz'})
    tx_cw_code: np.ndarray((20_000,), dtype=int) = field(
        default=np.load('_prn_code.npy'),
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': (20_000,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[chip0, ...] pseudo-random noise like code transmitted (contains full sequence)'})
    tx_chip_length: float = field(
        default=10e-6,
        metadata={'type': 'float64',
                  'units': 'second',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'chip length of pseudo-random code'})
    tx_code_length: float = field(
        default=10_000,
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'number of chips in the pseudo-random code'})
    rx_site_name: str = field(
        default='bakker',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 6,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of the transmitter site'})
    rx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
        default=np.asarray([52.243, -106.450]),
        metadata={'type': 'float64',
                  'units': 'degree',
                  'shape': (2,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[latitude, longitude] global North-Easting coordinates of the site in degrees'})
    rx_heading: float = field(
        default=7.0,
        metadata={'type': 'float64',
                  'units': 'degree',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'transmitter array boresight pointing direction in degrees East of North'})
    rx_rf_path: str = field(
        default='antenna->feedline->bulkhead->BPF->LNA->LNA->X300',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 48,
                  'version': __version__,
                  'created': __created__,
                  'description': 'RF hardware signal path chain string'})
    rx_ant_type: str = field(
        default='Cushcraft 50MHz Superboomer',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 27,
                  'version': __version__,
                  'created': __created__,
                  'description': 'brand and model of the antenna used'})
    rx_ant_coords: np.ndarray((3, 10), dtype=float) = field(
        default=np.asarray(
                       [[0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9],
                        [0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.],
                        [0., 0.0895, 0.3474, 0.2181, 0.6834, -0.0587, -1.0668, -0.7540, -0.5266, -0.4087]]),
        metadata={'type': 'float64',
                  'units': 'meter',
                  'shape': (3, 10),
                  'version': __version__,
                  'created': __created__,
                  'description': '[[x0, ...],[y0, ...],[z0, ...]] transmitter antenna locations in meters from antenna 0'})
    rx_feed_corr: np.ndarray = field(
        default=np.asarray(
                [[6.708204, 6.4031243, 6.0827622, 6.3245554, 6.4031243, 6.0827622, 6.708204, 6.0827622, 5.830952, 6.0],
                 [0.0, -13.95, -6.345, -5.89, -3.14, 16.86, 10.2, -1.25, 5.72, 3.015]]),
        metadata={'type': 'float64',
                  'units': 'volt, degree',
                  'shape': (2, 10),
                  'version': __version__,
                  'created': __created__,
                  'description': '[[ant0 magnitude, ...],[ant0 phase, ...]] feedline calibration correction per antenna'})
    rx_feed_corr_date: np.ndarray((3,), dtype=int) = field(
        default=np.asarray([0, 0, 0], dtype=int),
        metadata={'type': 'int32',
                  'units': 'year, month, day',
                  'shape': (3, ),
                  'version': __version__,
                  'created': __created__,
                  'description': '[year, month, day] date of the last feed line calibration correction'})
    rx_feed_corr_type: str = field(
        default='manual',
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 6,
                  'version': __version__,
                  'created': __created__,
                  'description': 'type of procedure used to measure calibration values (ex; manual, closure angle)'})
    rx_ant_mask: np.ndarray((10,), dtype=int) = field(
        default=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int),
        metadata={'type': 'int',
                  'units': 'None',
                  'shape': (10,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[ant0, ...] boolean mask (0 off, 1 on) indicating which antennas were used and/or functional'})
    rx_sample_rate: float = field(
        default=200_000.0,
        metadata={'type': 'float64',
                  'units': 'hertz',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the raw recorded sample rate at the receiver in Hz'})

    def __post_init__(self):
        # Todo: need to be careful about handling self.date
        self.time_resolution = self.tx_code_length * self.tx_chip_length
        self.coherent_integration_time = self.time_resolution * self.incoherent_averages


@dataclass(kw_only=True, slots=True)
class Data:
    # Typical time series aligned data
    time: np.ndarray
    rf_distance: np.ndarray
    snr_db: np.ndarray
    doppler_shift: np.ndarray
    spectra: np.ndarray
    spectra_variance: np.ndarray
    xspectra: np.ndarray
    xspectra_variance: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    slant_range: np.ndarray
    velocity_azimuth: np.ndarray
    velocity_elevation: np.ndarray
    velocity_magnitude: np.ndarray
    # Per second but for all range-Doppler bins
    avg_spectra_noise: np.ndarray
    spectra_noise: np.ndarray
    xspectra_noise: np.ndarray
    spectra_clutter_corr: np.ndarray
    xspectra_clutter_corr: np.ndarray
    data_flag: np.ndarray


@dataclass(kw_only=True, slots=True)
class Dev:
    raw_elevation: np.ndarray
    mean_jansky: np.ndarray
    max_jansky: np.ndarray
    validity: np.ndarray
    classification: np.ndarray
    azimuth_extent: np.ndarray
    elevation_extent: np.ndarray
    area: np.ndarray
    doppler_spectra: np.ndarray


@dataclass(kw_only=True, slots=True)
class Config:
    level0_dir: str = field(default='')
    level1_dir: str = field(default='')
    level2_dir: str = field(default='')
    level3_dir: str = field(default='')
    start_time: list[int] = field(default_factory=list)
    stop_time: list[int] = field(default_factory=list)
    step_time: list[int] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class Container:
    info: Info = field(default_factory=Info)
    data: Data = field(default_factory=Data)
    dev: Dev = field(default_factory=Dev)
    conf: Config = field(default_factory=Config)

    def __repr__(self):
        return f'info: {self.info!r}\n' \
               f'data: {self.data!r}\n' \
               f'dev: {self.dev!r}\n' \
               f'dev: {self.conf!r}'

    def show(self):
        msg = f'{"="*200}\n' \
              f'{"Dataset":^30} | ' \
              f'{"Units":^20} | ' \
              f'{"Type":^15} | ' \
              f'{"Shape":^15} | ' \
              f'{"Ver":^5} | ' \
              f'{"Created":^20} | ' \
              f'{"    Description":<}\n' \
              f'{"="*200}\n'
        for x in fields(self.info):
            msg += f'{"info."+x.name:<30} | ' \
                   f'{x.metadata["units"]:^20} | ' \
                   f'{x.metadata["type"]:^15} | ' \
                   f'{str(x.metadata["shape"]):^15} | ' \
                   f'{x.metadata["version"]:^5} | ' \
                   f'{x.metadata["created"]:^20} | ' \
                   f'{x.metadata["description"]:<}\n'
        for x in fields(self.data):
            msg += f'{"data."+x.name:<30} | ' \
                   # f'{x.metadata["units"]:^20} | ' \
                   # f'{x.metadata["type"]:^15} | ' \
                   # f'{str(x.metadata["shape"]):^15} | ' \
                   # f'{x.metadata["version"]:^5} | ' \
                   # f'{x.metadata["description"]:<}\n'
        for x in fields(self.dev):
            msg += f'{"dev."+x.name:<30} | ' \
                   # f'{x.metadata["units"]:^20} | ' \
                   # f'{x.metadata["type"]:^15} | ' \
                   # f'{str(x.metadata["shape"]):^15} | ' \
                   # f'{x.metadata["version"]:^5} | ' \
                   # f'{x.metadata["description"]:<}\n'
        return msg

    @classmethod
    def dataclass_to_hdf5(cls, path=''):
        f = h5py.File(path + 'temp.h5', 'w')

        def loop(k, n=''):
            for a in fields(k):
                key = a.name
                try:
                    value = getattr(k, a.name)
                except AttributeError as err:
                    if a.name in ['info', 'data', 'dev']:
                        pass
                    else:
                        raise AttributeError(f'Attribute {a.name} has no Value')
                if a.type is type:
                    dset = a.name + '/'
                    loop(value, dset)
                else:
                    if n != '':
                        if type(value) is str:
                            value = np.asarray(value, dtype='S')
                        else:
                            value = np.asarray(value)
                        print(n + key)
                        f.create_dataset(n + key, data=value)
                        for kee, vaa in a.metadata.items():
                            f[n + key].attrs[kee] = vaa

        loop(cls)

        f = h5py.File(path + 'temp.h5', 'r')
        print(f.items())

        def _print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f'\t{key}: {val}')
            return None

        f.visititems(_print_attrs)

        return

    def hdf5_to_dataclass(file: str):
        pass
        return


if __name__ == '__main__':
    d = Container()
    print(d.show())