try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False



def conj_kernel(arr1, arr2, averages, result):
    """
    Performs conjugate multiplication between two ndarrays ie: arr1 * conj(arr2)
    This is the cross-correlation of arr1 and arr2 in fourier space
    This function is performed assuming that multiple cross-correlations will be averaged together
    ----------
    arr1 : complex64 ndarray with shape (r+1, n)
        2D array [range gate, frequency] in fourier space. This array will NOT be conjugated
    arr2 : complex64 ndarray with shape (r+1, n)
        2D array [range gate, frequency] in fourier space. This array WILL be conjugated
    averages : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed
    result : complex64 ndarray with shape (r+1, n)
        2D array [range gate, frequency] in fourier space. The averaged results of the cross-correlation
        will be added to whatever is already in result and returned
    Returns
    -------
    result : complex64 ndarray with shape (r+1, n)
        2D array [range gate, frequency] containing the sum of its previous contents and the results of the cross correlation
    """
    xcorr = arr1*xp.conj(arr2)

    #todo: may need draven's floating point corrections

    result += xcorr/averages


    return result
