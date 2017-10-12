"""
This module implements some linear algebra utilities needed for data processing and prediction in PITECA.
"""


import numpy as np
import time
import scipy.signal as spsignal
from scipy.sparse.linalg import eigs


def demean(matrix, axis=0):
    if not axis:
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
    """
    adds a columns (or row) of ones to a matrix.
    :param matrix: a 2-d array
    :param axis: 1 for column, 0 for rows
    :return: the matrix with the added ones column/row
    """
    if axis==1:
        return np.concatenate((np.ones([np.size(matrix, 0), 1]), matrix), axis=1)
    if axis==0:
        return np.concatenate((np.ones([1, np.size(matrix, 1)]), matrix), axis=0)


def detrend(matrix, axis = 0):
    """
    removes a linear trend from the data.
    :param matrix: a 2-d matrix
    :param axis: the axis where to remove the trend
    :return: the matrix after detrending
    """
    return spsignal.detrend(matrix, axis)


def softmax(x, axis=0):
    """
    Compute the softmax function for each row or column of the input x.
    :param x: N dimensional vector or M x N dimensional numpy matrix.
    :param axis: 0 for rows, 1 for columns 
    :return: x after softmax
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

    else:
        # Vector
        v = np.exp(x - np.max(x))
        x = v/np.sum(v)

    if  x.shape != orig_shape:
        raise ValueError("shapes don't match")
    return x


def dim_0_dot_product(v1, v2):
    assert len(v1.shape) == len(v2.shape) == 1
    return v1.reshape([len(v1), 1]).dot(v2.reshape([1, len(v2)]))


def fsl_glm(x, y):
    """
    general linear model based on FSL's FEAT tool.
    To the purpose of this
    :param x: predictors matrix
    :param y: labels
    :return: a vector of t statistics for each predictor in x
    """
    beta = np.dot(np.linalg.pinv(x) , y)
    residuals = y - np.dot(x,beta)
    dof = np.size(y, 0) - np.linalg.matrix_rank(x)
    if dof<1 :
        raise ValueError("the rank of x is too small")
    sigma_sq = np.sum(residuals**2, axis=0) / dof
    varcope = dim_0_dot_product(np.diag(np.linalg.inv((x.transpose().dot(x)))), sigma_sq)
    t = beta / np.sqrt(varcope)
    return t


def bounded_svd_fast(x, n):
    """
    performs singular value decomposition to a 2d matrix x up to n components
    :param x: the input matrix of size MxN
    :param n: the max number of components
    :return: matrices u, s, v, such that:
            u is M x n
            s is  a diagonal matrix n x n
            v is n x N
            and u * s * v ~= x
    """
    x, x_t = (x, x.transpose()) if np.size(x,0) < np.size(x,1) else (x.transpose(), x)
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


def variance_normalise(x, thres = 2.3, n = 30, mindev = 0.001):
    """
    normalizes a matrix x according to the variance of the difference between x and 
    its approximation using bounded svd.
    :param x:  Txv matrix
    :param thres:  threshold
    :param n: num of singular values
    :param mindev: minimal value in stddev
    :return: x after variance normalization
    """


    start = time.time()
    uu, ss, vv = bounded_svd_fast(x, n)
    stop = time.time()
    print ("svd took {0:.3f} seconds.".format(stop-start))
    vv = np.transpose(vv)
    vv[abs(vv) < thres*np.std(vv, ddof=1)] = 0
    std_devs = np.std(x - np.matmul(np.matmul(uu, ss), vv.transpose()), axis=0, ddof=1)
    std_devs[std_devs<mindev] = mindev
    return x / std_devs


def rmse_loss(prediction, actual, use_normalization=False):
    """
    computes the root mean square error loss.
    each line in prediction is one observation
    """
    if not np.shape(prediction) == np.shape(actual):
        raise ValueError("prediction and actual must be the same shape")

    if use_normalization:
        prediction =  demean_and_normalize(prediction)
        actual = demean_and_normalize(actual)

    return np.sqrt(np.mean(np.square(prediction - actual)))

if __name__ == '__main__':
    x = np.random.random([10000, 90])
    n=30
    bounded_svd_fast(x, n)