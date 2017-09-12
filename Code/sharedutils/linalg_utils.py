import numpy as np
import time
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
    :param y:  Txv matrix
    :param thres:  threshold
    :param n: num of singular values?
    :param mindev: minimal value in stddev
    :return: y after variance normalization
    '''
    start = time.time()
    uu, ss, vv = ss_svds(y,n)
    stop = time.time()
    print ("svd took {0:.3f} seconds.".format(stop-start))
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


def detrend(matrix, axis = 0):
    return spsignal.detrend(matrix, axis)




def softmax(x, axis=0):
    """Compute the softmax function for each row of the input x.

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

def dim_0_dot(v1, v2):
    assert len(v1.shape) == len(v2.shape) == 1
    return v1.reshape([len(v1), 1]).dot(v2.reshape([1, len(v2)]))


def fsl_glm(x, y):

    beta = np.dot(np.linalg.pinv(x) , y)
    res = y - np.dot(x,beta)
    dof = np.size(y, 0) - np.linalg.matrix_rank(x)
    if dof<1 :
        raise ValueError("the rank of x is too small")
    sigma_sq = np.sum(res**2, axis=0) / dof;
    varcope = dim_0_dot(np.diag(np.linalg.inv((x.transpose().dot(x)))),  sigma_sq)
    t = beta / np.sqrt(varcope)
    return t


def rms_loss(prediction, actual, reduce_mean=False, use_normalization=False):
    """
    computes the residual sum of squares loss (using mean rather then sum)
    each line in prediction is one observation
    """
    if not np.shape(prediction) == np.shape(actual):
        raise ValueError("prediction and actual must be the same shape")

    if reduce_mean:
        prediction = demean(prediction, axis=1)
        actual = demean(actual, axis=1)

    if use_normalization:
        prediction = normalize(prediction, axis=1)
        actual = normalize(actual, axis= 1)

    rms = np.mean(np.square(prediction - actual))
    return rms
