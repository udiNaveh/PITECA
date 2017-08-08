import numpy as np
import time

from sharedutils.subject import *
from sharedutils.io_utils import *
from sharedutils.general_utils import *


def _get_task_maps_by_subject(subjects, task, getpath_func):
    task_maps = {}
    for subject in subjects:
        assert type(subject) == Subject

        subject_task_path = getpath_func(subject, task)
        #subject.predicted.get(task, None)
        if subject_task_path is None:
            # @error_handle - for development. In release we should guarantee that every subject
            # has the prediction?
            raise RuntimeWarning("no filepath found for subject {} for task {}".format(subject.subject_id, task.name))
        else:
            start = time.time()
            arr, (ax, bm) = open_cifti(subject_task_path)
            stop = time.time()
            print("opening {0} took {1:.4f} seconds".format(subject_task_path, stop-start))
            # @error_handle
            task_maps[subject] = arr
    return task_maps


def get_predicted_task_maps_by_subject(subjects, task):
    return _get_task_maps_by_subject(subjects, task, lambda s, t : s.predicted.get(task, None))


def get_actual_task_maps_by_subject(subjects, task):
    return _get_task_maps_by_subject(subjects, task, lambda s, t : s.actual.get(task, None))


def __arrays_to_matrix(arrays):
    arrays = list(arrays)
    if not all_same(arrays, lambda arr: np.shape(arr)):
        # @error_handle
        raise RuntimeWarning("not all files have the same brain model")

    if len(arrays[0].shape)==2:
        axis = 0 if np.size(arrays[0],axis=0)==1 else 1
        return np.concatenate(arrays, axis=axis)
    elif len(arrays[0].shape)==1:
        return np.stack(arrays, axis=0)
    else:
        # @error_handle
        raise Exception("cannot produce matrix for shape {}".format(arrays[0].shape))


def get_prediction_statistic(subjects, task, statfunc, outputpath = None):
    prediction_arrays = []
    for subject in subjects:
        assert type(subject)==Subject
        subject_predicted_task_path = subject.predicted.get(task, None)
        if subject_predicted_task_path is None:
            raise RuntimeWarning("no prediction for subject {0} for task {1}".format(subject.subject_id, task.name))
        else:
            arr, (ax, bm) = open_cifti(subject_predicted_task_path)
            prediction_arrays.append(arr)

    if not all_same(prediction_arrays, lambda arr : np.shape(arr)):
        raise RuntimeWarning("not all files have the same brain model")

    # need to verify that all arrays are of shape 1x59282
    res = statfunc(prediction_arrays)
    if outputpath is not None:
        save_to_dtseries(outputpath, bm, res)
    return res


def get_mean(arrays):
    return np.mean(np.concatenate(arrays, axis =0), axis=0)


def get_median(arrays):
    return np.median(np.concatenate(arrays, axis =0), axis=0)


def get_prediction_mean(subjects, task, outputpath):
    return get_prediction_statistic(subjects, task, get_mean ,outputpath)


def get_predictions_correlations(subjects, task, other_path):
    n_subjects = len(subjects)
    subjects_predictions = get_predicted_task_maps_by_subject(subjects, task)
    subjects_predictions_matrix = __arrays_to_matrix(
        [subjects_predictions[s] for s in subjects])

    mean_pred = np.mean(subjects_predictions_matrix, axis = 0)
    other_arr, (ax, bm) = open_cifti(other_path)

    # assume other_arr is of shape 1x91282

    unified_mat = np.concatenate((subjects_predictions_matrix,
                          mean_pred.reshape([1,STANDART_BM.N_CORTEX]),
                          other_arr[:,:STANDART_BM.N_CORTEX]))
    correlation_matrix = np.corrcoef(unified_mat)
    return (correlation_matrix[:n_subjects,:n_subjects], # subject x subject correlations
            correlation_matrix[:n_subjects,-2],  # correlations with group mean
        correlation_matrix[:n_subjects,-1]) # correlations with other_arr


def get_predicted_actual_correlations(subjects, task):
    '''
    
    :param subjects: type: List[Subject]
    :param task: type: Enum.Task
    :return:  a two-dimensional correlation matrix (nump.ndarray), in which the i,j
    entry is the pearson correlation between the prediction of the i'th subject and
    the actual activation of the j'th subject for task.
    '''

    subjects_predicted_maps = get_predicted_task_maps_by_subject(subjects, task)
    subjects_actual_maps = get_actual_task_maps_by_subject(subjects, task)

    # need to insure that maps are paired
    subjects = [s for s in subjects_predicted_maps if s in subjects_actual_maps]
    # @error_handel need to decide what to do if the above intersection is different from subjects
    # i.e. some subjects don't have actual and predicted maps
    n_subjects = len(subjects)
    predicted_matrix = __arrays_to_matrix([subjects_predicted_maps[s] for s in subjects])
    actual_matrix = __arrays_to_matrix([subjects_actual_maps[s] for s in subjects])

    correlation_matrix = np.corrcoef(actual_matrix[:, STANDART_BM.CORTEX],
                                     predicted_matrix[:, STANDART_BM.CORTEX])[n_subjects:, :n_subjects]
    return correlation_matrix

