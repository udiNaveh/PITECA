"""
This module was used for evaluation of models. It calculates a set of statistical measures 
about a model's prediction. This module is not part of the PITECA gui.
"""

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
from model import models, feature_extraction




def get_predictions_for_subjects_by_task(subjects, model):
    all_predictions = {}

    for i, subject in enumerate(subjects):
        all_predictions[subject.subject_id], _ =  model.predict(subject, save=False)
    all_predictions = inverse_dicts(all_predictions)

    return all_predictions



def evaluate_predictions(subjects, task_getter, model, predicted_by_task, include_all=True):
    """
    code for evaluating predictions of a given model.
    prints  
    :param subjects: the subjects for which the predictions were made.
    :param task_getter: an object of type MemTaskGetter that
    :param model: the model used for the prediction
    :param predicted_by_task: a dictionary that maps tasks to dictianries mapping subject ids to their predictions
    :param include_all: whether to use the prediction for the whole cortex in the evaluation or to disregard ~4500
                vertices that are not included in the original model.
    """

    roi_mask = np.load(definitions.ROI_MASK_PATH)
    if include_all:
        valid_veritices = np.ones(roi_mask.shape) > 0
    else:
        valid_veritices = roi_mask>0

    actual_by_task = {}
    for task in model.tasks:
        actual_by_task[task] = {s : task_getter.get_task_data(s, task) for s in subjects}
    tasks = model.tasks
    for task_index, task in enumerate(tasks):
        print("task: {0}".format(task.full_name))

        predicted_by_subj = {s : np.squeeze(predicted_by_task[task][s.subject_id])[valid_veritices] for s in subjects}
        actual_by_subj = {s: np.squeeze(actual_by_task[task][s])[valid_veritices] for s in subjects}

        #load canonical task activation:
        path_to_canonical = os.path.join(definitions.DATA_DIR, 'Tasks', 'canonical_{}.dtseries.nii'.format(task.full_name))
        canonical, (ax, bm) = open_1d_file(path_to_canonical)
        canonical = np.squeeze(canonical[0,:STANDARD_BM.N_CORTEX])[valid_veritices]
        canonical_normalized = demean_and_normalize(canonical)

        # get the rmse loss for every subject
        pred_act_losses = [rmse_loss(predicted_by_subj[s], actual_by_subj[s], use_normalization=True) for s in subjects]
        # get the predicted-actual correlation matrix
        all_corrs = get_predicted_actual_correlations(subjects, task, (predicted_by_subj, actual_by_subj))

        # for each subject - create maps with their difference from canonical activation
        actual_by_subj_diff_from_canonical = {s:demean_and_normalize(actual_by_subj[s]) - canonical_normalized for s in subjects}
        pred_by_subj_diff_from_canonical = {s: demean_and_normalize(predicted_by_subj[s]) - canonical_normalized for s in
                                              subjects}
        # rmse loss if canonical is predicted for every subject
        canonical_act_losses = [rmse_loss(canonical, actual_by_subj[s], use_normalization=True) for s in subjects]

        # how much the predictions a correlated with the canonical
        all_corrs_predicted_with_canonical  = \
            get_predicted_actual_correlations(subjects, task, ( {s : canonical for s in subjects},
                                                                    predicted_by_subj))

        # correlation between the predicted and actual individual difference from canonical
        pred_act_diffs_corrs = get_predicted_actual_correlations(subjects, task, (pred_by_subj_diff_from_canonical,
                                                                                  actual_by_subj_diff_from_canonical))

        # get the intersection over union between subjects' actual and predicted activation maps,
        # both for negative and positive significant activation areas

        iou_positive, iou_negative = get_significance_overlap_maps_for_subjects(subjects, task, None, subjects_predicted_maps=predicted_by_subj,
                                                   subjects_actual_maps=actual_by_subj, save=False)

        # get the intersection over union between subjects'  actual activation maps and the canonical,
        #  both for negative and positive significant actication areas.
        iou_positive_with_canonical, iou_negative_with_canonical = get_significance_overlap_maps_for_subjects(subjects, task, None,
                                                    subjects_predicted_maps={s : (np.squeeze(canonical))[:STANDARD_BM.N_CORTEX] for s in subjects},
                                                   subjects_actual_maps=actual_by_subj, save=False)

        # print statistics

        print("mean predicted-actual correlation with self = : {0:.3f}".format(np.mean(np.diag(all_corrs))))
        print("mean predicted-actual correlation with other = : {0:.3f}".
              format(np.mean([np.mean([all_corrs[i,j] for j in range(len(subjects)) if i!=j]) for i in range(len(subjects))])))

        print("mean corr of prediction with canonical  : {0:.3f}".
              format((np.mean(np.diag(all_corrs_predicted_with_canonical)))))
        print("median rmse with prediction : {0:.3f}".format(np.median(pred_act_losses)))
        print("median rmse with canonical : {0:.3f}".format(np.median(canonical_act_losses)))
        print("mean corr diffs : {0:.3f}".format(np.mean(np.diag(pred_act_diffs_corrs))))
        print("mean corr with other diff = : {0:.3f}"
              .format(np.mean([np.mean([pred_act_diffs_corrs[i,j] for j in range(len(subjects))
                                                                         if i!=j]) for i in range(len(subjects))])))

        print("iou positive with predictions: {0:.3f}\n iou positive with canonical: {1:.3f}".
              format(np.mean(iou_positive), np.mean(iou_positive_with_canonical)))
        print("iou negative with predictions: {0:.3f}\n iou negative with canonical: {1:.3f}".
              format(np.mean(iou_negative), np.mean(iou_negative_with_canonical)))



if __name__ == '__main__':
    all_features, all_tasks = load_data(normalize_features = False, normalize_tasks=False)
    subjects = [Subject(subject_id=zero_pad(i + 1, 6)) for i in range(200)]
    fe = feature_extraction.MemFeatureExtractor(all_features, subjects)
    modeltype = models.NN2lhModel # just for example, could be any Imodel
    tasks = [Task.MATH_STORY, Task.TWO_BK] # just for example, could be any avalable tasks
    tasks_getter = MemTaskGetter(all_tasks, subjects)
    subjects_training = subjects[:70]
    subjects_validation = subjects[70:100]
    subjects_test = subjects[100:200]
    model = models.model_factory(modeltype, tasks, feature_extractor = fe)
    tasks_getter = MemTaskGetter(all_tasks, subjects)
    predictions = get_predictions_for_subjects_by_task(subjects_test, model)
    evaluate_predictions(subjects_test, tasks_getter, model, predictions, include_all =False)