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





def get_predictions_for_subjects_by_task(subjects, model):
    all_predictions = {}

    for i, subject in enumerate(subjects):
        all_predictions[subject], _ =  model.predict(subject, save=False)
        print()
    all_predictions = inverse_dicts(all_predictions)
    return all_predictions



def get_correlations_for_subjects(subjects, task_getter, model, predictions=None, include_all=True):


    roi_mask = np.load(os.path.join(definitions.MODELS_DIR, 'roi_mask.npy'))
    if include_all:
        valid_veritices = np.ones(roi_mask.shape) >0
    else:
        valid_veritices = roi_mask>0

    predicted_by_task = get_predictions_for_subjects_by_task(subjects, model) if predictions is None else predictions
    save_pickle(predicted_by_task, os.path.join(definitions.LOCAL_DATA_DIR, '{}_learnedfrom100_pred.pkl'.format(model.__class__.__name__)))
    actual_by_task = {}
    for task in model.tasks:
        actual_by_task[task] = {s : task_getter.get_task_data(s, task) for s in subjects}
    tasks = model.tasks
    for task_index, task in enumerate(tasks):
        print("task = {0} : {1}".format(task_index, task.full_name))
        task_name = task

        loaded_subjects = []
        for s in subjects:
            predicted_by_task[task][s] = [predicted_by_task[task][so] for so in predicted_by_task[task].keys() if so.subject_id == s.subject_id][0]
        predicted_by_subj = {s : np.squeeze(predicted_by_task[task][s])[valid_veritices] for s in subjects}

        actual_by_subj = {s: np.squeeze(actual_by_task[task][s])[valid_veritices] for s in subjects}


        pred_act_losses = [rmse_loss(predicted_by_subj[s], actual_by_subj[s], use_normalization=True) for s in subjects]

        all_corrs = get_predicted_actual_correlations(subjects, task_name, (predicted_by_subj, actual_by_subj))
        np.save(os.path.join(definitions.MODELS_DIR, 'all_corrs_{0}_{1}.npy'.format(task.full_name, model.__class__.__name__)), all_corrs) #TODO
        path_to_canonical = os.path.join(definitions.DATA_DIR, 'Tasks', 'canonical_{}.dtseries.nii'.format(task.full_name))
        canonical, (ax, bm) = open_1d_file(path_to_canonical)
        canonical = np.squeeze(canonical[0,:STANDARD_BM.N_CORTEX])[valid_veritices]

        canonical_normalized = demean_and_normalize(canonical)
        actual_by_subj_diff_from_canonical = {s:demean_and_normalize(actual_by_subj[s]) - canonical_normalized for s in subjects}
        pred_by_subj_diff_from_canonical = {s: demean_and_normalize(predicted_by_subj[s]) - canonical_normalized for s in
                                              subjects}

        canonical_act_losses = [rmse_loss(canonical, actual_by_subj[s], use_normalization=True) for s in subjects]
        all_corrs_with_canonical  = \
            get_predicted_actual_correlations(subjects, task_name, ( {s : canonical for s in subjects},
                                                                    actual_by_subj))
        all_corrs_predicted_with_canonical  = \
            get_predicted_actual_correlations(subjects, task_name, ( {s : canonical for s in subjects},
                                                                    predicted_by_subj))



        pred_act_diffs_losses = [rmse_loss(pred_by_subj_diff_from_canonical[s], actual_by_subj_diff_from_canonical[s], use_normalization=False) for s in subjects]
        pred_act_diffs_corrs = get_predicted_actual_correlations(subjects, task_name, (pred_by_subj_diff_from_canonical, actual_by_subj_diff_from_canonical))

        all_corrs_normalized = demean_and_normalize(demean_and_normalize(all_corrs, axis=0), axis=1)
        print("mean corr with other = : {0:.3f}".format(np.mean([np.mean([all_corrs[i,j] for j in range(len(subjects))
                                                                         if i!=j]) for i in range(len(subjects))])))
        print("mean corr with self = : {0:.3f}".format(np.mean(np.diag(all_corrs))))
        print("mean corr with canonical  : {0:.3f}".format((np.mean(np.diag(all_corrs_with_canonical)))))

        print("mean corr of prediction with canonical  : {0:.3f}".format((np.mean(np.diag(all_corrs_predicted_with_canonical)))))
        print("mean self corrs normalized  : {0:.3f}".format(np.mean(np.diag(all_corrs_normalized))))
        print("median rmse with prediction : {0:.3f}".format(np.median(pred_act_losses)))
        print("median rmse with canonical : {0:.3f}".format(np.median(canonical_act_losses)))
        print("median rmse diffs : {0:.3f}".format(np.median(pred_act_diffs_losses)))
        print("mean corr diffs : {0:.3f}".format(np.mean(np.diag(pred_act_diffs_corrs))))
        print("mean corr with other diff = : {0:.3f}".format(np.mean([np.mean([pred_act_diffs_corrs[i,j] for j in range(len(subjects))
                                                                         if i!=j]) for i in range(len(subjects))])))


        iou_positive, iou_negative = get_significance_overlap_maps_for_subjects(subjects, task, None, subjects_predicted_maps=predicted_by_subj,
                                                   subjects_actual_maps=actual_by_subj, save=False)

        #todo delete
        #save_pickle((all_corrs, iou_positive, iou_negative, subjects), os.path.join(definitions.LOCAL_DATA_DIR, 'stats{}_nn.pkl'.format(task.full_name)))

        iou_positive_2, iou_negative_2 = get_significance_overlap_maps_for_subjects(subjects, task, None,
                                                    subjects_predicted_maps={s : (np.squeeze(canonical))[:STANDARD_BM.N_CORTEX] for s in subjects},
                                                   subjects_actual_maps=actual_by_subj, save=False)
        print("iou positive with predictions: {0:.3f}\n iou positive with canonical: {1:.3f}".format(np.mean(iou_positive), np.mean(iou_positive_2)))
        print("iou negative with predictions: {0:.3f}\n iou negative with canonical: {1:.3f}".format(np.mean(iou_negative), np.mean(iou_negative_2)))

    return all_corrs


def run_evaluation(model):


    model = models.model_factory(model, TASKS, fe)

    tasks_getter = MemTaskGetter(all_tasks, subjects)
    subjects_training = subjects[:70]
    subjects_validation = subjects[70:100]
    subjects_test = subjects[130:200]
    #predictions = open_pickle(os.path.join(definitions.LOCAL_DATA_DIR, 'NN2lhModelWithFiltersAndTaskAsFeaturesPredictions.pkl'))
    get_correlations_for_subjects(subjects_test, tasks_getter, model)

if __name__ == '__main__':
    all_features, all_tasks = load_data(normalize_features = False, normalize_tasks=False)
    subjects = [Subject(subject_id=zero_pad(i + 1, 6)) for i in range(200)]
    fe = models.MemFeatureExtractor(all_features, subjects)

    #for modeltype in [models.TFLinearFSF, models.TFLinear, models.NN2lhModelWithFiltersAsFeatures, models.NN2lhModel, models.TFLinearAveraged]:
    for modeltype in [models.NN2lhModelWithFiltersAsFeatures,
                      ]:
        print(modeltype.__name__)
        #preds_path = os.path.join(r'D:\Projects\PITECA\Data\all predictions mat','predictions_{}_70_first.pkl'.format(modeltype.__name__))
        #preds = open_pickle(preds_path)
        model = models.model_factory(modeltype, TASKS, fe)
        subjects_test = subjects[100:200]
        tasks_getter = MemTaskGetter(all_tasks, subjects)
        get_correlations_for_subjects(subjects_test, tasks_getter, model, include_all =False)
        tf.reset_default_graph()
        print()
