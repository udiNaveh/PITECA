import numpy as np
import time
from sklearn.preprocessing import normalize
import scipy.signal as spsignal
import scipy.stats.mstats as mstats
from scipy.sparse.linalg import eigs

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
    uu, ss, vv = ss_svds_fast(y,n)
    stop = time.time()
    print ("svd took {0:.3f} seconds.".format(stop-start))
    vv = np.transpose(vv)
    vv[abs(vv) < thres*np.std(vv, ddof=1)] = 0
    std_devs = np.std(y - np.matmul(np.matmul(uu,ss), vv.transpose()), axis=0, ddof=1)
    std_devs[std_devs<mindev] = mindev
    return y / std_devs


def ss_svds(x, n):

    u, s, v = np.linalg.svd(x, full_matrices =False)
    if (np.size(s)) > n:
        s = s[:n]
        u = u[:, :n]
        v = v[:n, :]

    s = np.diag(s)

    return (u,s,v)

def ss_svds_fast(x, n):

    x, x_t = (x, x.transpose()) if np.size(x,0) < np.size(x,1) else (x.transpose, x)
    x_squared = np.dot(x, x_t)
    if n < np.size(x,0):
        eigenvalues, eigenvectors = eigs(x_squared, n)
        eigenvalues = eigenvalues.astype('float')
        eigenvectors = eigenvectors.astype('float')
    else:
        eigenvalues, eigenvectors = np.linalg.eig(x_squared)
    eigenvalues_inds = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues_inds[::-1]]
    eigenvectors = eigenvectors[:, eigenvalues_inds[::-1]]
    s = np.sqrt(np.abs(eigenvalues))
    v = (np.matmul(x_t, np.dot(eigenvectors, np.diag(1/s)))).transpose()

    return (eigenvectors,np.diag(s),v)

def demean(matrix, axis=0):
    if axis == 0:
        return matrix-np.mean(matrix, axis=axis)
    if axis >0:
        return matrix-(np.mean(matrix, axis=axis, keepdims=True))


def demean_and_normalize(matrix, axis=0):
    demeaned = demean(matrix, axis=axis)
    std = np.std(demeaned, axis= axis, keepdims=True)
    try:
        normalized = demeaned / std
    except FloatingPointError:
        print("err")
    return normalized


def add_ones_column(matrix, axis=1):
    # matrix is a 2-d array
    if axis==1:
        return np.concatenate((np.ones([np.size(matrix, 0), 1]), matrix), axis=1)
    if axis==0:
        return np.concatenate((np.ones([1, np.size(matrix, 1)]), matrix), axis=0)

def detrend(matrix, axis = 0):
    return spsignal.detrend(matrix, axis)


def z_score(array, axis=0, ddof=0):
    return mstats.zscore(array, axis, ddof)

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

    if  x.shape != orig_shape:
        raise ValueError("shapes don't match")
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


def rmse_loss(prediction, actual, reduce_mean=False, use_normalization=False):
    """
    computes the root mean square error loss.
    each line in prediction is one observation
    """
    if not np.shape(prediction) == np.shape(actual):
        raise ValueError("prediction and actual must be the same shape")

    #if len(np.shape(prediction))==1:
    #prediction = np.reshape(prediction, [1, len(prediction)])
    #if len(np.shape(actual))==1:
    #actual = np.reshape(actual, [1, len(actual)])

    if reduce_mean:
        prediction = demean(prediction, axis=1)
        actual = demean(actual, axis=1)

    if use_normalization:
        prediction =  z_score(prediction, axis=1)
        actual = z_score(actual, axis= 1)

    return np.sqrt(np.mean(np.square(prediction - actual)))