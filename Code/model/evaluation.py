import numpy as np
import os
import pickle

import definitions
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.subject import *
from analysis.analyzer import *
from sharedutils.io_utils import *
from model.data_manager import *
from model import models



TASKS = [Task.MATH_STORY,
                   Task.TOM,
                   Task.MATCH_REL,
                   Task.TWO_BK,
                   Task.REWARD,
                   Task.FACES_SHAPES,
                   Task.T
                   ]


def get_predictions_for_subjects_by_task(subjects, model):
    all_predictions = {}

    for i, subject in enumerate(subjects):
        all_predictions[subject], _ =  model.predict(subject, save=False)
        print()
    all_predictions = inverse_dicts(all_predictions)
    return all_predictions


def get_correlations_for_subjects(subjects, task_getter, model, predictions=None):

    predicted_by_task = get_predictions_for_subjects_by_task(subjects, model) if predictions is None else predictions
    actual_by_task = {}
    for task in model.tasks:
        actual_by_task[task] = {s : task_getter.get_task_data(s, task) for s in subjects}
    tasks = model.tasks
    for task_index, task in enumerate(tasks):
        print("task = {0} : {1}".format(task_index, task.full_name))
        task_name = task
        predicted_by_subj = {s : np.squeeze(predicted_by_task[task][s]) for s in subjects}
        actual_by_subj = {s: np.squeeze(actual_by_task[task][s]) for s in subjects}

        all_corrs = get_predicted_actual_correlations(subjects, task_name, (predicted_by_subj, actual_by_subj))
        path_to_canonical = os.path.join(definitions.DATA_DIR, 'Tasks', 'canonical_{}.dtseries.nii'.format(task.full_name))
        canonical, (ax, bm) = open_1d_file(path_to_canonical)

        all_corrs_with_canonical  = \
            get_predicted_actual_correlations(subjects, task_name, ( {s : (np.squeeze(canonical))[:STANDARD_BM.N_CORTEX] for s in subjects},
                                                                    actual_by_subj))
        all_corrs_normalized = demean_and_normalize(demean_and_normalize(all_corrs, axis=0), axis=1)
        print("mean corr with other = : {0:.3f}".format(np.mean([np.mean([all_corrs[i,j] for j in range(len(subjects))
                                                                         if i!=j]) for i in range(len(subjects))])))
        print("mean corr with self = : {0:.3f}".format(np.mean(np.diag(all_corrs))))
        print("mean corr with canonical = : {0:.3f}".format((np.mean(np.diag(all_corrs_with_canonical)))))
        print("mean self corrs normalized = : {0:.3f}".format(np.mean(np.diag(all_corrs_normalized))))
        predicted_mean_s = [np.mean(predicted_by_task[task][s]) for s in subjects]
        iou_positive, iou_negative = get_significance_overlap_maps_for_subjects(subjects, task, None, subjects_predicted_maps=predicted_by_subj,
                                                   subjects_actual_maps=actual_by_subj, save=False)

        iou_positive_2, iou_negative_2 = get_significance_overlap_maps_for_subjects(subjects, task, None,
                                                    subjects_predicted_maps={s : (np.squeeze(canonical))[:STANDARD_BM.N_CORTEX] for s in subjects},
                                                   subjects_actual_maps=actual_by_subj, save=False)
        print("iou positive with predictions: {0:.3f}, iou positive with canonical: {1:.3f}".format(np.mean(iou_positive), np.mean(iou_positive_2)))
        print("iou negative with predictions: {0:.3f}, iou negative with canonical: {1:.3f}".format(np.mean(iou_negative), np.mean(iou_negative_2)))

    return all_corrs


def run_evaluation(model):

    all_features, all_tasks = load_data()
    subjects = [Subject(subject_id=zero_pad(i + 1, 6)) for i in range(200)]
    fe = models.MemFeatureExtractor(all_features, subjects)
    model = models.model_factory(model, TASKS, fe)
    #model = models.NN2lhModelWithFiltersAbdMeanAsFeatures([Task.MATH_STORY], fe, means = np.mean(all_tasks, axis=1))
    tasks_getter = MemTaskGetter(all_tasks, subjects)
    subjects_training = subjects[:70]
    subjects_validation = subjects[70:100]
    subjects_test = subjects[130:]
    get_correlations_for_subjects(subjects_test, tasks_getter, model)



if __name__ == '__main__':

    run_evaluation(models.NN2lhModelWithFiltersAndTaskAsFeatures)