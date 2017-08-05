import numpy as np
from sklearn.preprocessing import normalize
import scipy.signal as spsignal

'''
Specific implementations of linear algebra utilities for processing our data.
Includes mainly (or only) translation from matlab to python of functions used in
Ido's code. These methods are used for preprocessing and feature extraction.

'''


def variance_normalise(y, thres = 2.3, n = 30, mindev = 0.001):

    '''  
      documentation
    :param y:  Txv threshold
    :param n: num of sungular values?
    :param mindev: minimal value in stddev
    :return: y after variance normalization
    '''
    uu, ss, vv = ss_svds(y,n)
    vv = np.transpose(vv)
    vv[abs(vv) < thres*np.std(vv, ddof=1)] = 0
    std_devs = np.std(y - np.matmul(np.matmul(uu,ss), vv.transpose()), axis=0, ddof=1)
    std_devs[std_devs<mindev] = mindev
    return y / std_devs


def ss_svds(y,n):

    u, s, v = np.linalg.svd(y, full_matrices =False)
    if (np.size(s)) > n:
        s = s[:n]
        u = u[:, :n]
        v = v[:n, :]

    s = np.diag(s)

    return (u,s,v)


def demean(matrix, axis=0):
    if axis == 0:
        return matrix-np.mean(matrix, axis=axis)
    if axis == 1:
        return matrix-(np.mean(matrix, axis=axis))[:,np.newaxis]



def demean_and_normalize(matrix, axis=0):
    return normalize(demean(matrix, axis=axis),'l2',axis=axis)


def detrend(matrix):
    return spsignal.detrend(matrix, 0)

def softmax(x, axis=0):
    """Compute the softmax function for each row of the input x.

   

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) == 2:
        # Matrix
        M,N = x.shape
        if axis == 0:
            v = np.exp(x - np.max(x,1).reshape((M,1)))
            x = v/np.sum(v,1).reshape((M,1))
        elif axis == 1:
            v = np.exp(x - np.max(x,0).reshape((1,N)))
            x = v/np.sum(v,0).reshape((1,N))
        ### END YOUR CODE
    else:
        # Vector
        v = np.exp(x - np.max(x))
        x = v/np.sum(v)

    assert x.shape == orig_shape
    return x
