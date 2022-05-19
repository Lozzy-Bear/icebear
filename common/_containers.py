from dataclasses import dataclass, field, fields
import numpy as np
import h5py


# Info
@dataclass
class Prelate:
    tx_site_name: str = field(
        default='prelate',
        metadata={'type': 'str',
                  'units': 'alphabet',
                  'shape': '',
                  'version': '',
                  'description': 'name of the transmitter site'})
    tx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
        default=np.asarray([50.893, -109.403]),
        metadata={'type': 'ndarray[float]',
                  'units': 'degrees',
                  'shape': (2,),
                  'version': '',
                  'description': 'global North-Easting [latitude, longitude] coordinates of the site in degrees'})
    tx_heading: float = field(
        default=16.0,
        metadata={'type': 'float',
                  'units': 'degrees',
                  'shape': (1,),
                  'version': '',
                  'description': 'transmitter array boresight pointing direction in degrees East of North'})
    tx_rf_path: str = field(
        default='X300->amplifier->bulkhead->feedline->antenna',
        metadata={'type': 'str',
                  'units': 'alphabet',
                  'shape': '',
                  'version': '',
                  'description': 'RF hardware signal path chain string'})
    tx_ant_type: str = field(
        default='Cushcraft A50-5S',
        metadata={'type': 'str',
                  'units': 'alphabet',
                  'shape': '',
                  'version': '',
                  'description': 'brand and model of the antenna used'})
    tx_ant_coords: np.ndarray((3, 10), dtype=float) = field(
        default=np.asarray([[0., 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
        metadata={'type': 'ndarray[float]',
                  'units': 'meters',
                  'shape': (3, 10),
                  'version': '',
                  'description': '[[x0, ...],[y0, ...],[z0, ...]] transmitter antenna locations in meters from antenna 0'})
    tx_feed_corr: np.ndarray = field(
        default=np.asarray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
        metadata={'type': 'ndarray[float]',
                  'units': 'volts, degrees',
                  'shape': (2, 10),
                  'version': '',
                  'description': '[[ant 0 magnitude, ...],[ant 0 phase, ...]] feedline calibration correction per antenna'})
    tx_feed_corr_date: np.ndarray((3,), dtype=int) = field(
        default=np.asarray([0, 0, 0]),
        metadata={'type': 'ndarray[float]',
                  'units': 'years, months, days',
                  'shape': (3, ),
                  'version': '',
                  'description': '[year, month, day] date of the last feed line calibration correction'})
    tx_feed_corr_type: str = field(
        default='manual',
        metadata={'type': 'str',
                  'units': 'alphabet',
                  'shape': '',
                  'version': '',
                  'description': 'type of procedure used to measure calibration values (ex; manual, closure angle)'})
    tx_ant_mask: np.ndarray((10,), dtype=int) = field(
        default=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        metadata={'type': 'ndarray[float]',
                  'units': '',
                  'shape': (10,),
                  'version': '',
                  'description': '[ant 0, ...] boolean mask (0 off, 1 on) indicating which antennas were used and/or functional'})
    tx_sample_rate: float = field(
        default=800_000.0,
        metadata={'type': 'float',
                  'units': 'Hz',
                  'shape': (1,),
                  'version': '',
                  'description': 'sample rate of transmitted code in Hz'})
    tx_cw_code: np.ndarray((20_000,), dtype=int) = field(
        default=np.load('_prn_code.npy'),
        metadata={'type': 'ndarray[int]',
                  'units': '',
                  'shape': (20_000,),
                  'version': '',
                  'description': 'pseudo-random noise like code transmitted (contains full sequence)'})


@dataclass
class Bakker:
    rx_site_name: str = field(default='bakker', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'name of the transmitter site'})
    rx_site_lat_lon: np.ndarray((2,), dtype=float) = field(default=np.asarray([52.243, -106.450]), metadata={
        'type': 'ndarray[float]',
        'units': 'degrees',
        'shape': (2,),
        'version': '',
        'description': 'global North-Easting [latitude, longitude] coordinates of the site in degrees'})
    rx_heading: float = field(default=7.0, metadata={
        'type': 'float',
        'units': 'degrees',
        'shape': (1,),
        'version': '',
        'description': 'transmitter array boresight pointing direction in degrees East of North'})
    rx_rf_path: str = field(default='antenna->feedline->bulkhead->BPF->LNA->LNA->X300', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'RF hardware signal path chain string'})
    rx_ant_type: str = field(default='Cushcraft 50MHz Superboomer', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'brand and model of the antenna used'})
    rx_ant_coords: np.ndarray((3, 10), dtype=float) = field(default=np.asarray(
                       [[0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9],
                        [0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.],
                        [0., 0.0895, 0.3474, 0.2181, 0.6834, -0.0587, -1.0668, -0.7540, -0.5266, -0.4087]]), metadata={
        'type': 'ndarray[float]',
        'units': 'meters',
        'shape': (3, 10),
        'version': '',
        'description': '[[x0, ...],[y0, ...],[z0, ...]] transmitter antenna locations in meters from antenna 0'})
    rx_feed_corr: np.ndarray = field(default=np.asarray(
                [[6.708204, 6.4031243, 6.0827622, 6.3245554, 6.4031243, 6.0827622, 6.708204, 6.0827622, 5.830952, 6.0],
                 [0.0, -13.95, -6.345, -5.89, -3.14, 16.86, 10.2, -1.25, 5.72, 3.015]]), metadata={
        'type': 'ndarray[float]',
        'units': 'volts, degrees',
        'shape': (2, 10),
        'version': '',
        'description': '[[ant 0 magnitude, ...],[ant 0 phase, ...]] feedline calibration correction per antenna'})
    rx_feed_corr_date: np.ndarray((3,), dtype=int) = field(default=np.asarray([0, 0, 0]), metadata={
        'type': 'ndarray[float]',
        'units': 'years, months, days',
        'shape': (3, ),
        'version': '',
        'description': '[year, month, day] date of the last feed line calibration correction'})
    rx_feed_corr_type: str = field(default='manual', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'type of procedure used to measure calibration values (ex; manual, closure angle)'})
    rx_ant_mask: np.ndarray((10,), dtype=int) = field(default=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), metadata={
        'type': 'ndarray[float]',
        'units': '',
        'shape': (10,),
        'version': '',
        'description': '[ant 0, ...] boolean mask (0 off, 1 on) indicating which antennas were used and/or functional'})
    rx_sample_rate: float = field(default=200_000.0, metadata={
        'type': 'float',
        'units': 'Hz',
        'shape': (1,),
        'version': '',
        'description': 'the raw recorded sample rate at the receiver in Hz'})


@dataclass
class Info:
    experiment_name: str = field(default='normal', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'name of experiment ran (ex; normal, mobile)'})
    radar_config: str = field(default='ib3d', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'name of the radar the data was collected with (ex; ib, ib3d, lofar)'})
    center_freq: float = field(default=49_500_000.0, metadata={
        'type': 'float',
        'units': 'Hz',
        'shape': '',
        'version': '',
        'description': 'radar center frequency in Hz'})


# Data
# All data levels should become time synced arrays per day. We can use temp.h5 in between stages
@dataclass(order=True)
class Level1Data:
    info = Info
    itemB: float = 70.0


@dataclass(order=True)
class Level2Data:
    item2: float = 3232.12
    item3: str = 'poop'
    item1: float = 4.0


@dataclass(order=True)
class Level3Data:
    item1: int = 6
    item2: float = 70.0


@dataclass()
class Packer:
    @classmethod
    def _to_hdf5(cls, path=''):
        f = h5py.File(path + 'temp.h5', 'w')

        def loop(k, n=''):
            for a in fields(k):
                key = a.name
                try:
                    value = getattr(k, a.name)
                except AttributeError as err:
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

    def _to_dataclass(file: str):
        pass
        return


@dataclass()
class Validator:
    def _validate_range(self, key, value):
        min = self.__dataclass_fields__[key].metadata['min']
        max = self.__dataclass_fields__[key].metadata['max']
        if value and not min <= value <= max:
            raise ValueError(f'Attribute {key}:{value} is not between {min} and {max}')

    def _validate_choices(self, key, value):
        options = self.__dataclass_fields__[key].metadata['options']
        if options and value not in options:
            raise ValueError(f'Attribute {key}:{value} is not an expected option:{options}')

    def _validate_type(self, key, value):
        type = self.__dataclass_fields__[key].default_factory
        if not isinstance(value, type):
            raise ValueError(f'Attribute {key}:{type(value)} is not the expected type:{type}')

    def _validate_shape(self, key, value):
        shape = self.__dataclass_fields__[key].metadata['shape']
        if value.shape is not shape:
            raise ValueError(f'Attribute {key}{value.shape} is not the expected shape:{shape}')


@dataclass()
class Container(Validator, Packer):
    info: type
    data: type

    def __init__(self, tx, rx, level):
        self.info = Info(tx, rx)
        self.data = level


if __name__ == '__main__':
    # bcode = generate_bcode('/home/arl203/icebear/icebear/dat/pseudo_random_code_test_8_lpf.txt')
    # print(bcode.shape, bcode.size * 32 / (8*1e3), 'kB')
    # np.save('normal_prn_code', bcode)
    d = Container(Prelate, Bakker, Level1Data)
    print(d.info)
    d._to_hdf5()

    # x = Level2Data
    # y = Level3Data
    # print(x.__annotations__, y.__annotations__)
    # for attr in x.__annotations__.items():
    #     if attr in y.__annotations__.items():
    #         print(attr, 'here')
    #     elif attr not in y.__annotations__.items():
    #         print(attr, 'missing')
    # _to_hdf5(x, '')