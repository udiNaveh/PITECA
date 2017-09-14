import cifti
import numpy as np
from sharedutils.constants import *
import sharedutils.cmd_utils as cmd_utils
import sharedutils.asynch_utils as asynch_utils
import sharedutils.cmd_utils
import scipy.io as sio
import h5py

'''
All methods that read or write to files.
These are mainly wrapper functions for third-party io functions 
(implemented in the cifti package, or other packages we will use).
This can allow us handle exception in a certain way or add some needed processing stages
to data. It can also make other code more readable.
Other methods for reading or saving files can be added here - for example saving graphs created
in the analysis module.



'''


def open_cifti(path):
    '''
    This is not properly handled yet
    :param path:
    :return: arr, (axis, bm)
    '''
    try:
        return cifti.read(path)
    except FileNotFoundError:
        pass
        # @error_handle
        return None

    except ValueError as e:
        if str(e) == 'Only CIFTI-2 files are supported':
            # @error_handle
            answer, cifti2path = asynch_utils.do_convert_to_CIFTI2(path)
            if answer==False:
                # write to logger that the file could not be opened because it was not cifti2
                return None
            else:
                success = cmd_utils.convert_to_CIFTI2(path, cifti2path)
                # not actually a boolean. need to change implementation, maybe use try-except.
                if success:
                    return open_rfmri_file(cifti2path)
    # other exceptions?


def open_rfmri_file(path):

    arr, (series, bm) = open_cifti(path)
    if not isinstance(series, cifti.axis.Series):
        raise ValueError("input file is not a time series")
    if np.size(arr, 1) < MIN_TIME_UNITS:
        raise ValueError("Input file must include at least {0} time units".format(MIN_TIME_UNITS))
    return arr, (series, bm)


def open_features_file(path):
    arr, (series, bm) = open_cifti(path)
    if not isinstance(series, cifti.axis.Series):
        raise ValueError("input file is not a time series")
    if np.size(arr, 1) != NUM_FEATURES:
        raise ValueError("features file must include {} features".format(NUM_FEATURES))
    return arr, (series, bm)


def save_to_dtseries(filename, brain_model, mat, fill_with_zeros = False):
    '''
    save a dtseries.nii file given the data matrix and the brain model.
    The series axis is generated so as to fit the size of the matrix (every row is a time point).


    :param filename: the file to save
    :param brain_model: a Cifti.Axis.BrainModel object
    :param mat: the data matrix
    :param fill_with_zeros: havn't implemented yet. The idea is that if we make prediction for only
      the cortex vertices, we can make a brain model that contains only the cortex vertices
      and save to a file like that (save disc space). On the other hand someone (like Ido) might want to have
      all the 91282 vertices and just have zeros in the unused vertices.
    :return:
    '''
    if len(np.shape(mat)) == 2:
        assert mat.shape[1] == np.size(brain_model.arr)
    elif len(np.shape(mat)) == 1:
        mat = np.reshape(mat, [1, np.size(brain_model.arr)])
    series = cifti.Series(start=0, step=1, size=mat.shape[0])
    if not filename.endswith(".dtseries.nii"):
        filename += ".dtseries.nii"
    cifti.write(filename, mat, (series, brain_model))
    return filename


def save_to_dscalar(filename, brain_model, mat, names, zeropad = False):
    raise NotImplementedError





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

#TODO DEKETE
def get_subject_to_feature_index_mapping(path):
    mapping = {}
    with open(path, 'r') as f:
        for i in range(100):
            subj_number = int(f.readline()) + 1
            assert subj_number not in mapping
            mapping[subj_number] = i
    return mapping
