import numpy as np
import os
import pickle

import definitions
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.subject import *
from analysis.analyzer import get_predicted_actual_correlations

import matplotlib.pyplot as plt

tasks_file = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', "moreTasks.npy")
all_tasks_200s_new_path = r'D:\Projects\PITECA\Data_for_testing\time_series\allTasksReordered.npy'
all_features_path_200 = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order_200.npy")
all_features_path = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order.npy")


TASKS = [Task.MATH_STORY,
                   Task.TOM,
                   Task.MATCH_REL,
                   Task.TWO_BK,
                   Task.REWARD,
                   Task.FACES_SHAPES,
                   Task.T,
                   ]

def load_data(features_path = all_features_path_200, tasks_path = all_tasks_200s_new_path):
    all_tasks = np.load(tasks_path, mmap_mode='c')
    all_tasks = all_tasks[:STANDARD_BM.N_CORTEX,:,:]
    all_features = np.load(features_path, mmap_mode='c')
    assert np.size(all_features, axis=0) == np.size(all_tasks, axis=0)
    assert np.size(all_features, axis=1) == np.size(all_tasks, axis=1)
    assert np.size(all_features, axis=2) == NUM_FEATURES
    all_features = demean_and_normalize(all_features[:,:,:], axis=0)
    all_tasks = demean_and_normalize(all_tasks[:,:,:], axis=0)
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




#
# if __name__ == '__main__':
#     features, tasks = load_data(all_features_path_200, all_tasks_200s_new_path)