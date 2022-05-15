from dataclasses import dataclass, field, fields
import types
import numpy as np
import h5py


# Info
@dataclass
class Transmitter:
    tx_heading: float = 16.0


@dataclass
class Receiver:
    rx_heading: float = 7.0


@dataclass
class General:
    gen: float = 7.0


@dataclass
class Settings:
    set: float = 7.0


@dataclass
class Info(Transmitter, Receiver, General, Settings):
    pass


# Data
# All data levels should become time synced arrays per day. We can use temp.h5 in between stages
@dataclass(order=True)
class Level1Data:
    itemA: int = 6
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


# Products
@dataclass
class Level1:
    info: classmethod = Info
    data: classmethod = Level1Data
    desc: str = 'test'
    tstr: np.ndarray = np.arange(10)#, dtype=np.complex128)


@dataclass
class Level2:
    info: classmethod = Info
    data: classmethod = Level2Data
    # dev: classmethod = Dev


@dataclass
class Level3:
    info: classmethod = Info
    data: classmethod = Level3Data
    # dev: classmethod = Dev


def _to_hdf5(cls: dataclass, path=''):
    f = h5py.File(path + 'temp.h5', 'w')

    def loop(k, n=''):
        for a in fields(k):
            key = a.name
            value = getattr(k, a.name)
            if a.type is classmethod:
                dset = a.name + '/'
                loop(value, dset)
            else:
                if type(value) is str:
                    value = value#np.asarray(value, dtype='S')
                else:
                    value = np.asarray(value)
                f.create_dataset(n + key, data=value)
                print(n + key, value, type(value))
    loop(cls)

    print('--------------h5py load')
    f.visititems(_print_attrs)
    j = f['desc'][...]
    print(f'this {j}')
    return


def _print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return None


def _to_dataclass(file: str):
    pass
    return


if __name__ == '__main__':
    c = Level1
    _to_hdf5(c, '')

    # x = Level2Data
    # y = Level3Data
    # print(x.__annotations__, y.__annotations__)
    # for attr in x.__annotations__.items():
    #     if attr in y.__annotations__.items():
    #         print(attr, 'here')
    #     elif attr not in y.__annotations__.items():
    #         print(attr, 'missing')
    # _to_hdf5(x, '')