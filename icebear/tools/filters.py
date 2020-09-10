import numpy as np


def meteor():
    """
    Determines which level 1 data indices are likely to contain meteors.

    Parameters
    ----------


    Returns
    -------
        indices : int np.array
            Array of index values of level 1 data which contains meteors.

    Notes
    -----
        Meteors can be classified in level 1 data by the following conditions
        * Doppler shift near 0 Hz
        * Extremely narrow spectral width
        * Meteors may exist in all ranges
        *

        Objective; feed in level 1 data, detect where meteors are, pass those indices to imaging.

    """
    return indices
