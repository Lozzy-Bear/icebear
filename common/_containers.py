from dataclasses import dataclass, field, fields
import numpy as np
import h5py


# Info
@dataclass
class Transmitter:
    tx_site_name: str = field(default='prelate', metadata={
        'type': 'str',
        'units': 'alphabet',
        'shape': '',
        'version': '',
        'description': 'name of the transmitter site'})
    tx_site_lat_lon: list[float] = field(default=lambda: [50.893, -109.403], metadata={
        'type': 'list[float]',
        'units': 'degrees',
        'shape': (2,),
        'version': '',
        'description': 'global North-Easting [latitude, longitude] coordinates of the site in degrees'
    })
    tx_heading: float = field(default=7.0, metadata={
        'type': 'float',
        'units': 'degrees',
        'shape': (1,),
        'version': '',
        'description': 'transmitter array boresight pointing direction in degrees East of North'
    })


@dataclass
class Receiver:
    rx_heading: float = 7.0


@dataclass
class Settings:
    set: float = 7.0


@dataclass
class Info(Transmitter, Receiver):
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
    center_freq: float = field(default=49500000.0, metadata={
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
                    if type(value) is str:
                        value = np.asarray(value, dtype='S')
                    else:
                        value = np.asarray(value)
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
    info: type = Info
    data: type = Level1Data


if __name__ == '__main__':
    d = Container
    d.info.tx_site_name = 'prelate'
    d.info.tx_site_lat_lon = [12.3, -120.2]
    # print(d.info.__annotations__)
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