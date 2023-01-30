try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False

def windowed_view(ndarray, window_len, step):
    """
    Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
    otherwise be dropped without missing samples needed for the convolutions windows. The
    strides will also not extend out of bounds meaning we do not need to pad extra samples and
    then drop bad samples after the fact.
    :param      ndarray:     The input ndarray
    :type       ndarray:     ndarray
    :param      window_len:  The window length(filter length)
    :type       window_len:  int
    :param      step:        The step(dm rate)
    :type       step:        int
    :returns:   The array with a new view.
    :rtype:     ndarray
    """

    nrows = ((ndarray.shape[-1] - window_len) // step) + 1
    last_dim_stride = ndarray.strides[-1]
    new_shape = ndarray.shape[:-1] + (nrows, window_len)
    new_strides = list(ndarray.strides + (last_dim_stride,))
    new_strides[-2] *= step

    return xp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)

def ssmf_kernel(meas, code, N=20000, r=2000, dec_rate=200):
    """
    Performs matched filtering and decimation on a set of complex measurements. Uses Cupy to run on the GPU
    Parameters
    ----------
    meas : complex64 ndarray with shape (N + r)
        1D array of measurements
    code : complex64 ndarray with shape (N)
        1D array of PRN code values
    N : int
        length of PRN code in samples (default 20000)
    r : int
        number of range gates (default 2000)
    dec_rate : int
         rate at which to decimate the values in meas array (default 200)
    Returns
    -------
    result : complex64 ndarray with shape (r+1, N/dec_rate)
        2D array [range gate, time] of the results of matched filtering and decimation
    """
    # todo: test on real data/ against draven's results
    # todo: look for a better way to make the input_samples array besides the for-loop
    # decimation and matched filtering
    # result[i, j] = xp.sum(meas[j*dec_rate+i:(j+1)*dec_rate+i)] * xp.conj(code[j*dec_rate:(j+1)*dec_rate]))
    result = xp.ndarray((r+1, int(N/dec_rate)), xp.complex64)

    input_samples = xp.ndarray((int(N/dec_rate), r+1, dec_rate), xp.int)
    for i in range(0, int(N/dec_rate)):
        # don't love this. if it's allocating more memory for the arrays, not very good. double window somehow instead?
        entry = windowed_view(meas[i*dec_rate:r + (i + 1)*dec_rate], window_len=dec_rate, step=1)
        input_samples[i, :, :] = entry

    code_samples = windowed_view(code, window_len=dec_rate, step=dec_rate)

    # assuming N=20000, r=2000, dec_rate=200:
    #
    # input_samples= [[[    meas[0:200]    ],
    # (100x2001x200)   [    meas[1:201]    ],
    #                          ...
    #                  [  meas[2000:2200]  ]],
    #
    #                 [[   meas[200:400]   ],
    #                  [   meas[201:401]   ],
    #                          ...
    #                  [  meas[2200:2400]  ]],
    #
    #                          ...
    #
    #                 [[ meas[19800:20000] ],
    #                  [ meas[19801:20001] ],
    #                          ...
    #                  [ meas[21800:22000] ]]]
    #
    #
    #  code_samples = [[    code[0:200]    ],
    #  (100x200)       [   code[200:400]   ],
    #                          ...
    #                  [ code[19800:20000] ]]

    # result[i, j] = xp.sum(input_samples[j*dec_rate + i] * xp.conj(code_samples[j]))
    result = xp.einsum('ijk,ik->ji', input_samples, xp.conj(code_samples))

    return result

# N = 20000
# r = 2000
# dec_rate = 200
# meas = xp.arange(0, N+r)
# code = xp.arange(0, N)

# print(ssmf_kernel(meas, code, N, r, dec_rate))


# these two operations should give same result:

# A = xp.array(  [[[0, 1, 0],
#                  [1, 1, 0],
#                  [1, 1, 1]],
#                 [[0, 2, 0],
#                  [2, 2, 0],
#                  [2, 2, 2]],
#                 [[0, 5, 0],
#                  [5, 5, 0],
#                  [5, 5, 5]]])

# B = xp.array([[2, 3, 1], [1, 5, 0], [4, 4, 3]])
#
# print(xp.einsum('ijk,jk->ij', A, B))
#
# A = xp.array(  [[[0, 1, 0],
#                  [0, 2, 0],
#                  [0, 5, 0]],
#                 [[1, 1, 0],
#                  [2, 2, 0],
#                  [5, 5, 0]],
#                 [[1, 1, 1],
#                  [2, 2, 2],
#                  [5, 5, 5]]])
#
# B = xp.array([[2, 3, 1], [1, 5, 0], [4, 4, 3]])
#
# print(xp.einsum('ijk,ik->ji', A, B))