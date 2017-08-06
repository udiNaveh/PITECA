import numpy as np

from sharedutils.subject import *
from sharedutils.io_utils import *
from sharedutils.general_utils import *


def get_subjects_task_predictions_matrix(subjects, task):
    prediction_arrays = []
    for subject in subjects:
        assert type(subject) == Subject

        subject_predicted_task_path = subject.predicted.get(task, None)
        if subject_predicted_task_path is None:
            raise RuntimeWarning("no prediction for subject {} for task {}".format(subject.subject_id, task.name))
        else:
            arr, (ax, bm) = open_cifti(subject_predicted_task_path)
            prediction_arrays.append(arr)

    if not all_same(prediction_arrays, lambda arr: np.shape(arr)):
        raise RuntimeWarning("not all files have the same brain model")

    if len(prediction_arrays[0].shape)==2:
        axis = 0 if np.size(prediction_arrays[0],axis=0)==1 else 1
        return np.concatenate(prediction_arrays, axis=axis)
    elif len(prediction_arrays[0].shape)==1:
        return np.stack(prediction_arrays, axis=0)
    else:
        raise Exception("cannot produce prediction matrix for shape {}".format(prediction_arrays[0].shape))


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
    preds = get_subjects_task_predictions_matrix(subjects, task)
    mean_pred = np.mean(preds, axis = 0)
    other_arr, (ax, bm) = open_cifti(other_path)

    # assume other_arr is of shape 1x91282

    mat = np.concatenate((preds, mean_pred.reshape([1,STANDART_BM.N_CORTEX]), other_arr[:,:STANDART_BM.N_CORTEX]))
    correlation_matrix = np.corrcoef(mat)
    return (correlation_matrix[:n_subjects,:n_subjects], correlation_matrix[:n_subjects,-2],
        correlation_matrix[:n_subjects,-1])





if __name__ == "__main__":

    create_subjects()
    print(correlation_matrix.shape)