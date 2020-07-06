import numpy as np
import ctypes as C
import pyfftw
import numba
pyfftw.interfaces.cache.enable()


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    """
    Computes the square of input value for real and imaginary portions of complex values.

    Args:
        x (complex np.array): Complex values.

    Returns:
        return (complex np.array): Squared real and imaginary components.
    """

    return x.real**2 + x.imag**2


def func():
    """ 
    Python wrapper for GPU CUDA code in the libssmf.so file.

    Args:
        *

    Returns:
       func (wrapper): Python function wrapper with inputs identical to wrapped CUDA function in the ssmf.cu file.

    Noes:
       * Wrapped function is ssmf.cu is denoted by extern "C" {} tag
       * Inputs are (cufftComplex *meas1, cufftComplex *meas2, cufftComplex *code, cufftComplex *result, size_t measlen, size_t codelen, size_t size, ing avg, ing check)
    """

    dll = C.CDLL('./libssmf.so', mode=C.RTLD_GLOBAL)
    func = dll.ssmf
    func.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_size_t,
                     C.c_size_t, C.c_size_t, C.c_int, C.c_int]
    return func


"""
    Links wrapped CUDA function to be calleable as python function
   
    Notes:
        * May be alternative ways to perform C/CUDA wrapping
"""
__fmed = func()


def ssmf(meas, code, averages, nrang, fdec, codelen):
    """
    Formats measured data and CUDA function inputs and calls wrapped function to determine single antenna spectra.

    Args:
        meas (complex64 np.array): Antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 np.array): Transmitted psuedo-random code sequence.
        averages (int): The number of 0.1 second averages to be performed on the GPU.
        nrang (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S    (complex64 np.array): 2D Spectrum output for antenna (Doppler shift x Range).

    Notes:
        * ssmf.cu could be modified to allow code to be a float32 input. This would reduce memory requirements
          on the GPU.
        * 'check' input of __fmed is 0 which indicates a single antenna is being processed.
    """

    nfreq = int(codelen / fdec)
    measlen = codelen * averages + nrang
    result_size = nfreq * nrang

    code = code.astype(np.complex64)
    result = np.zeros((nrang, nfreq), dtype=np.complex64)

    m_p = meas.ctypes.data_as(C.POINTER(C.c_float))
    c_p = code.ctypes.data_as(C.POINTER(C.c_float))
    r_p = result.ctypes.data_as(C.POINTER(C.c_float))

    __fmed(m_p, c_p, r_p, measlen, codelen, result_size, averages, 0)

    S = abs2(result)
    return S


def ssmfx(meas0, meas1, code, averages, nrang, fdec, codelen):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        meas0 (complex64 np.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        meas1 (complex64 np.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 np.array): Transmitted psuedo-random code sequence.
        averages (int): The number of 0.1 second averages to be performed on the GPU.
        nrang (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).

    Notes:
        * ssmf.cu could be modified to allow code to be a float32 input. This would reduce memory requirements
          on the GPU.
        * 'check' input of __fmed is 1 which indicates a pair of antennas is being processed.
    """

    measlen0 = len(meas0)
    measlen1 = len(meas1)
    nfreq = int(codelen / fdec)
    result_size = nfreq * nrang

    code = code.astype(np.complex64)
    result = np.zeros((nrang, nfreq), dtype=np.complex64)

    m_p0 = meas0.ctypes.data_as(C.POINTER(C.c_float))
    m_p1 = meas1.ctypes.data_as(C.POINTER(C.c_float))
    c_p = code.ctypes.data_as(C.POINTER(C.c_float))
    r_p = result.ctypes.data_as(C.POINTER(C.c_float))

    __fmed(m_p0, m_p1, c_p, r_p, measlen0, codelen, result_size, averages, 1)

    S = result
    return S
