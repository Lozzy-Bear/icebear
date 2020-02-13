import ctypes as C
import numpy as n
import time
import numba
import pyfftw
pyfftw.interfaces.cache.enable()

# to compile the library: 
# gcc -shared -fPIC -O3 -o libssmf.so ssmf.c

# install openblas and pyfftw
@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

# more efficient c operations

# Stuff to set up access to C routine for median filter (fmedfil)
_fmed = n.ctypeslib.load_library('libssmf', './Ccode')
_fmed.ssmf.restype = C.c_int
_fmed.ssmf.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]
def ssmf(meas, code):
    # these are hardocded in the C program to make it compile to faster code (vectorization)
    nrange=2000
    dec=200
    codelen=20000
    nt=codelen/dec
    result=n.zeros((nrange,codelen/dec),dtype=n.complex64)
    _fmed.ssmf(meas.ctypes.data_as(C.POINTER(C.c_float)), code.ctypes.data_as(C.POINTER(C.c_float)),result.ctypes.data_as(C.POINTER(C.c_float)))
    S=(pyfftw.interfaces.scipy_fftpack.fft(result))
    return(S)

def ssmfx(meas0, meas1, code):
    # these are hardocded in the C program to make it compile to faster code (vectorization)
    nrange=2000
    dec=200
    codelen=20000
    nt=codelen/dec
    result0=n.zeros((nrange,codelen/dec),dtype=n.complex64)
    result1=n.zeros((nrange,codelen/dec),dtype=n.complex64)

    _fmed.ssmf(meas0.ctypes.data_as(C.POINTER(C.c_float)), code.ctypes.data_as(C.POINTER(C.c_float)),result0.ctypes.data_as(C.POINTER(C.c_float)))
    _fmed.ssmf(meas1.ctypes.data_as(C.POINTER(C.c_float)), code.ctypes.data_as(C.POINTER(C.c_float)),result1.ctypes.data_as(C.POINTER(C.c_float)))

    S=pyfftw.interfaces.scipy_fftpack.fft(result0)*n.conj(pyfftw.interfaces.scipy_fftpack.fft(result1))
    return(S)
if __name__ == "__main__":
    nrange=2000
    codelen=20000
    dec=200
    meas=n.zeros(codelen+nrange,dtype=n.complex64)
#    result=n.zeros((nrange,codelen/dec),dtype=n.complex64)
    code=n.zeros(codelen,dtype=n.float32)
    for i in range(100):
        t0=time.time()
        # hardcoded codelen and reslen to vectorize
        S=ssmf(meas,code)
        t1=time.time()
        print("dt %1.2f"%((t1-t0)/0.1))
