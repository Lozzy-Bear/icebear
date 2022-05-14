import os
import pathlib
from yachalk import chalk

import h5py
import dataclasses
import numpy as np
try:
    import cupy as xp
except ModuleNotFoundError as err:
    import numpy as xp
    print(chalk.red('Failed: ') + chalk.green(os.path.basename(__file__)) + ' failed to import ' +
          chalk.green('cupy') + ' -- importing ' + chalk.green('numpy'))


def data2h5(cls, file):
    """
    Convert a dataclass to an HDF5 file.

    Parameters
    ----------
    cls : class.object
        Dataclass object holding processed data to be stored in level 1 or level 2 HDF5 format.
    file : str
        Filename and path to the output HDF5 file

    Returns
    -------
        None
    """

    return None

@dataclasses.dataclass
class Test:
    d1: np.ndarray = np.arange(1000)
    d2: list = dataclasses.field(default=list['str1', 'str2'])
    d3: int = 1
    d4: str = 'str3'

C = Test

print(C.d1)

