"""
Methods used for loading and handling data in model learning/evaluation.
Not used in the workflows accessible from PITECA GUI. 
"""

import numpy as np
import os
import pickle
import definitions
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.subject import Subject
from analysis.analyzer import get_predicted_actual_correlations
import matplotlib.pyplot as plt

TASKS_MATRIX_PATH = '' # REPLACE WITH YOUR OWN
FEATURES_MATRIX_PATH = '' # REPLACE WITH YOUR OWN
# TODO delete
TASKS_MATRIX_PATH = r'D:\Projects\PITECA\Data_for_testing\time_series\allTasksReordered.npy'
FEATURES_MATRIX_PATH = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order_200.npy")


TASKS = AVAILABLE_TASKS

def load_data(features_path = FEATURES_MATRIX_PATH, tasks_path = TASKS_MATRIX_PATH, normalize_features = True, normalize_tasks = True):
    all_tasks = np.load(tasks_path, mmap_mode='c')
    all_tasks = all_tasks[:STANDARD_BM.N_CORTEX,:,:]
    all_features = np.load(features_path, mmap_mode='c')
    assert np.size(all_features, axis=0) == np.size(all_tasks, axis=0)
    assert np.size(all_features, axis=1) == np.size(all_tasks, axis=1)
    assert np.size(all_features, axis=2) == NUM_FEATURES
    if normalize_features:
        all_features = demean_and_normalize(all_features[:,:,:], axis=0)
    if normalize_tasks:

        all_tasks = demean_and_normalize(all_tasks[:,:,:], axis=None)
    return all_features, all_tasks


class MemTaskGetter:

    def __init__(self, tasks_mat, subjects, subjects_mapping=None, tasks_mapping=None):
        self.tasks_mat = tasks_mat
        self.subjects_mapping = subjects_mapping if subjects_mapping is not None else \
            {subject : int(subject.subject_id)-1 for subject in subjects}
        self.tasks_mapping = tasks_mapping if tasks_mapping is not None else \
            {task : i for i,task in enumerate(TASKS)}


    def get_task_data(self, subjects, tasks, roi_indices=None):
        subjects = [subjects] if isinstance(subjects, Subject) else subjects
        tasks = [tasks] if isinstance(tasks, Task) else tasks
        subj_idx = [self.subjects_mapping[subj] for subj in subjects]
        subjects_bool_idx = [i in subj_idx for i in range(np.size(self.tasks_mat, axis=1))]
        task_idx = [self.tasks_mapping[task] for task in tasks]
        task_bool_idx = [i in task_idx for i in range(np.size(self.tasks_mat, axis=2))]

        task_data = self.tasks_mat[:, subjects_bool_idx, :][:,:,task_bool_idx]
        if roi_indices is not None:
            task_data = task_data[roi_indices,:,:]
        return task_data



def get_subjects_features_from_matrix(mat, subjects, roi_indices = None, mapping=None):
    if not mapping:
        mapping = {subject : int(subject.subject_id)-1 for subject in subjects}
    subj_idx = [mapping[subj] for subj in subjects]
    mask = [i in subj_idx for i in range(np.size(mat,axis=1))]

    if roi_indices is None:
        return mat[:, mask, :]
    else:
        return mat[:, mask, :][roi_indices,:,:]


def get_selected_features_and_tasks(individual_features_matrix, subjects, roi_indices, task, memtaskgetter, global_features_matrix=None):
    roi_features = get_subjects_features_from_matrix(individual_features_matrix, subjects, roi_indices)
    if global_features_matrix is not None:
        roi_feats = np.concatenate([np.concatenate((roi_features[:,i,:],
                                                    global_features_matrix[roi_indices, :]), axis=1)
                                    for i in range(len(subjects))], axis=0)
    else:
        roi_feats = np.concatenate([roi_features[:,i,:] for i in range(len(subjects))], axis=0)

    roi_task = memtaskgetter.get_task_data(subjects, task, roi_indices)
    roi_task = np.concatenate([roi_task[:,i,:] for i in range(len(subjects))], axis=0)

    return roi_feats, roi_task