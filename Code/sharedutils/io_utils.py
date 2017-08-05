import cifti
import numpy as np
from sharedutils.constants import *
import sharedutils.cmd_utils as cmd_utils
import sharedutils.asynch_utils as asynch_utils
import sharedutils.cmd_utils

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
        # do something
        return None

    except ValueError as e:
        if str(e) == 'Only CIFTI-2 files are supported':
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


def save_to_dtseries(filename, brain_model, mat, zeropad = False):
    '''
    save a dtseries.nii file given the data matrix and the brain model.
    The series axis is generated so as to fit the size of the matrix (every row is a time point).
    
    
    :param filename: the file to save 
    :param brain_model: a Cifti.Axis.BrainModel object
    :param mat: the data matrix
    :param zeropad: havn't implemented yet. The idea is that if we make prediction for only
      the cortex vertices, we can make a brain model that contains only the cortex vertices
      and save to a file like that (save disc space). On the other hand someone (like Ido) might want to have 
      all the 91282 vertices and just have zeros in the unused vertices.
    :return: 
    '''
    assert len(np.shape(mat)) == 2
    assert mat.shape[1] == brain_model.arr.size()
    series = cifti.Series(start=0, step=1, size=mat.shape[0])
    cifti.write(filename, mat, (series, brain_model))
    return True

def save_to_dscalar(filename, brain_model, mat, names, zeropad = False):

    pass