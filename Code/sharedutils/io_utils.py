"""
Methods for reading or writing to files. These are mainly wrapper functions for third-party io functions 
(e.g. implemented in the cifti package), that allow exception handling for the diefferent scenarios used
in PITECA.
"""

import cifti
import numpy as np
from sharedutils.constants import *
import sharedutils.cmd_utils as cmd_utils
import scipy.io as sio
import h5py
import pickle
import os
from tempfile import TemporaryDirectory
import definitions
from definitions import PitecaError
import GUI.globals as gb


def open_cifti(path):
    """
    A wrapper to the cifti.read method, that also allows handling files not in cifti-2 format 
    (using wb_command convert_to_CIFTI2 command)
    :param path: a path to a .nii file
    :return: arr, (axis, bm) - see cifti.read documentation
    """
    try:
        return cifti.read(path)

    except ValueError as ve:
        if str(ve) == 'Only CIFTI-2 files are supported':
            with TemporaryDirectory() as tmp_dir:
                cifti2path = os.path.join(tmp_dir, os.path.basename(path))
                cmd_utils.convert_to_CIFTI2(path, cifti2path)
                arr, (ax, bm) = cifti.read(cifti2path)
                arr2 = np.copy(arr)
                del arr
                return arr2, (ax, bm)
    except FileNotFoundError as fnfe:
        raise PitecaError(fnfe.strerror, path)


def open_dtseries(path):
    arr, (series, bm) = open_cifti(path)
    if not isinstance(series, cifti.axis.Series):
        raise PitecaError("Input file is not a time series", path)
    if np.size(arr, 1) not in (STANDARD_BM.N_CORTEX, STANDARD_BM.N_TOTAL_VERTICES):
        raise PitecaError("Data in file does not match standard brain model", path)
    return arr, (series, bm)


def open_rfmri_input_data(path):
    arr, (series, bm) = open_dtseries(path)
    if np.size(arr, 1) != STANDARD_BM.N_TOTAL_VERTICES:
        raise PitecaError("Data in file must include a full standard brain model ({} vertices).".format(STANDARD_BM.N_TOTAL_VERTICES), path)
    if np.size(arr, 0) < MIN_TIME_UNITS:
        raise PitecaError("Input file must include at least {0} time units.".format(MIN_TIME_UNITS), path)
    return arr, (series, bm)


def open_features_file(path):
    arr, (series, bm) = open_dtseries(path)
    if np.size(arr, 0) != NUM_FEATURES:
        raise PitecaError("features file must include {} features".format(NUM_FEATURES), path)
    if np.size(arr, 1) > STANDARD_BM.N_CORTEX:
        arr = arr[:,:STANDARD_BM.N_CORTEX]
    return arr, (series, bm)


def open_1d_file(path):
    arr, (series, bm) = open_dtseries(path)
    if np.size(arr, 0) > 1:
        raise PitecaError("Input file is in 2D, expected 1D data", path)
    return arr, (series, bm)


def save_to_dtseries(filename, brain_model, mat):
    '''
    saves a dtseries.nii file given the data matrix and the brain model.
    The series axis is generated so as to fit the size of the matrix (every row is a time point).

    :param filename: the file to save
    :param brain_model: a Cifti.Axis.BrainModel object
    :param mat: the data matrix
    :return:
    '''
    if len(np.shape(mat)) == 2:
        assert mat.shape[1] == np.size(brain_model.arr)
    elif len(np.shape(mat)) == 1:
        mat = np.reshape(mat, [1, np.size(brain_model.arr)])
    series = cifti.Series(start=0, step=1, size=mat.shape[0])
    if not filename.endswith(".dtseries.nii"):
        filename += ".dtseries.nii"
    gb.curr_cifti_filenames.append(filename)
    cifti.write(filename, mat, (series, brain_model))
    return filename


def get_bm(bm_type):
    if bm_type == 'full':
        bm = pickle.load(open(definitions.BM_FULL_PATH, 'rb'))
        return bm
    elif bm_type == 'cortex':
        bm = pickle.load(open(definitions.BM_CORTEX_PATH, 'rb'))
        return bm
    else:
        raise ValueError('BM type ""{}"" does not exist'.format(bm_type))


def open_mat_file(filepath):
    arrays = {}
    try:
        mat = sio.loadmat(filepath)
        print("file {} loaded".format(filepath))
        for key, val in mat.items():
            if isinstance(val, np.ndarray):
                arrays[key] = val
    except NotImplementedError:
        f = h5py.File(filepath)
        print("file {} loaded".format(filepath))
        for k, v in f.items():
            arrays[k] = np.array(v)
    return arrays


def load_ndarray_from_mat(filepath, array_name = None):
    arrays = open_mat_file(filepath)
    if len(arrays)!=1:
        if array_name in arrays:
            return arrays[array_name]
        raise ValueError("file must include only one array")
    else:
        return arrays.popitem()[1]


def safe_open(path, *args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, *args)

def open_pickle(path):
    return pickle.load(safe_open(path, 'rb'))

def save_pickle(obj, path):
    return pickle.dump(obj, safe_open(path, 'wb'))





