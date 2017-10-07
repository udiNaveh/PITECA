import numpy as np
import os
import pickle

import definitions
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.subject import *
from analysis.analyzer import get_predicted_actual_correlations
from model import models

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
                   Task.T
                   ]


class MemTaskGetter:

    def __init__(self, tasks_mat, subjects_mapping, tasks_mapping):
        self.tasks_mat = tasks_mat
        self.subjects_mapping = subjects_mapping
        self.tasks_mapping = tasks_mapping

    def get_task(self,subject, task):
        subj_idx = self.subjects_mapping[int(subject.subject_id)]
        tasks_idx = self.tasks_mapping[task]
        task_raw = self.tasks_mat[tasks_idx, subj_idx, :]
        return task_raw


def get_predictions_for_subjects_by_task(subjects, model):
    all_predictions = {}

    for i, subject in enumerate(subjects):
        all_predictions[subject], _ =  model.predict(subject, save=False)
        print()
    all_predictions = inverse_dicts(all_predictions)
    return all_predictions


def get_correlations_for_subjects(subjects, task_getter, model, predictions=None):

    predicted_by_task = get_predictions_for_subjects_by_task(subjects, model) if not predictions else predictions
    actual_by_task = {}
    for task in model.tasks:
        actual_by_task[task] = {s : task_getter.get_task(s, task) for s in subjects}
    tasks = model.tasks
    for task_index, task in enumerate(tasks):
        print("task = {0} : {1}".format(task_index, task.full_name))
        task_name = task
        predicted_by_subj = {s : np.squeeze(predicted_by_task[task][s]) for s in subjects}
        actual_by_subj = {s: actual_by_task[task][s] for s in subjects}
        all_corrs = get_predicted_actual_correlations(subjects, task_name, (predicted_by_subj, actual_by_subj))
        all_corrs_normalized = demean_and_normalize(demean_and_normalize(all_corrs, axis=0), axis=1)
        print("mean corr with other = : {0:.3f}".format(np.mean([np.mean([all_corrs[i,j] for j in range(len(subjects))
                                                                         if i!=j]) for i in range(len(subjects))])))
        print("mean corr with self = : {0:.3f}".format(np.mean(np.diag(all_corrs))))
        print("mean self corrs normalized = : {0:.3f}".format(np.mean(np.diag(all_corrs_normalized))))
    return


def run_evaluation(model_name):
    print('running evaluation for model: ' + model_name)
    all_tasks = np.load(all_tasks_200s_new_path)
    all_tasks = all_tasks[:,:,:STANDART_BM.N_CORTEX]
    all_features = np.load(all_features_path_200)
    subjects_mapping = {i+1 :i for i in range(200)}
    tasks_mapping = {t : idx for idx, t in enumerate(TASKS)}
    tasks_getter = MemTaskGetter(all_tasks, subjects_mapping, tasks_mapping)
    fe = models.MemFeatureExtractor(all_features, subjects_mapping)
    model = models.model_factory(model_name, TASKS, fe)
    subjects = [Subject(subject_id=zeropad(i+1, 6)) for i in range(200)]
    subjects_training = subjects[:70]
    subjects_validation = subjects[70:100]
    subjects_test = subjects[100:130]
    get_correlations_for_subjects(subjects_test, tasks_getter, model)



if __name__ == '__main__':

    run_evaluation('fing')
    run_evaluation('MLP by ROI with group connectivity features')