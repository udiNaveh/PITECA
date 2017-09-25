import numpy as np
import tensorflow as tf
import os
import random
import pickle
import matplotlib.pyplot as plt
import cifti
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.io_utils import *
from sharedutils.subject import *
from sharedutils.general_utils import safe_open
from model.models import *

import sharedutils.general_utils as general_utils
import definitions

from misc.nn_model import *
from misc.model_hyperparams import *
from analysis.analyzer import get_predicted_actual_correlations


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

LINEAR_BETAS_PATH = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'linear', 'loo_betas_7_tasks_take2')
NN_WEIGHTS_PATH = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'nn')
# file names
spatial_filters_file = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', "spatial_filters.npy")
tasks_file = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', "moreTasks.npy")
spatial_filters_path = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')
subjects_features_order = os.path.join(definitions.LOCAL_DATA_DIR, 'subjects_features_order.txt')
all_features_path = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order.npy")

fsf_nn_pred_path = r'D:\Projects\PITECA\Data\model\nn\predictions\predictions_validation_s71-100_nn_2hl_fsf.npy'
nn_pred_path = r'D:\Projects\PITECA\Data\model\nn\predictions\predictions_validation_s71-100_nn_2hl.npy'
linear_pred_path = r'D:\Projects\PITECA\Data\model\linear\loo_betas_7_tasks_take2\linear_weights_70_s_2\predictions_validation_s71-100_linear.npy'
linear_pred_path_fsf = r'D:\Projects\PITECA\Data\model\linear\loo_betas_7_tasks_take2\linear_weights_70_s_2\predictions_validation_s71-100_linear_fsf.npy'

TASKS = {Task.MATH_STORY: 0,
                   Task.TOM: 1,
                   Task.MATCH_REL: 2,
                   Task.TWO_BK: 3,
                   Task.REWARD: 4,
                   Task.FACES_SHAPES: 5,
                   Task.T: 6
                   }


def load_data():
    all_features_raw = np.load(all_features_path)
    spatial_filters_raw, (series, bm) = cifti.read(spatial_filters_path)
    spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDART_BM.CORTEX])
    soft_filters = softmax((spatial_filters_raw).astype(float) * TEMPERATURE)
    soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
    soft_filters[:,2] = 0
    soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDART_BM.N_CORTEX, 1])
    hard_filters = np.round(softmax((spatial_filters_raw).astype(float) * 1000))
    hard_filters[spatial_filters_raw<SPATIAL_FILTERS_THRESHOLD] = 0
    all_features_normalized = demean_and_normalize(all_features_raw, axis=0)
    all_tasks = np.load(tasks_file)
    all_tasks = all_tasks[:,:, :STANDART_BM.N_CORTEX]
    return all_features_normalized, all_tasks, soft_filters, hard_filters, spatial_filters_raw

all_features_normalized, all_tasks, soft_filters, hard_filters, spatial_filters_raw= load_data()
all_tasks_normalized = demean_and_normalize(all_tasks, axis=2)
n_filters = np.size(hard_filters, axis=1)
spatial_filters_raw = demean_and_normalize(spatial_filters_raw, axis=0)
n_additional_features = n_filters if USE_RAW_FILTERS_AS_FEATURES else 0


nn_scope_name = 'nn1_h2_reg'
tensors_two_hidden_layer_nn =\
    regression_with_two_hidden_layers_build(NUM_FEATURES+n_additional_features, 1, nn_scope_name)
x, y, y_pred = tensors_two_hidden_layer_nn
loss_function = build_loss(y, y_pred, nn_scope_name, REG_LAMBDA, HUBER_DELTA)



def get_subject_features_from_matrix(subject):
    subj_features = all_features_normalized[: STANDART_BM.N_CORTEX, int(subject.subject_id) -1, :]
    if USE_RAW_FILTERS_AS_FEATURES:
        subj_features = np.concatenate((subj_features, spatial_filters_raw), axis = 1)
    return subj_features


def get_selected_features_and_tasks(all_features, ind, tasks, subjects):
    if USE_RAW_FILTERS_AS_FEATURES:
        roi_feats = np.concatenate([np.concatenate((all_features[ind, int(s.subject_id) - 1, :],
                                                    spatial_filters_raw[ind, :]), axis=1)
                                    for s in subjects], axis=0)

    else:
        roi_feats = np.concatenate([all_features[ind, int(s.subject_id)-1, :] for s in subjects], axis=0)

    roi_tasks = np.concatenate([tasks[:, int(s.subject_id) - 1, ind] for s in subjects], axis=1)
    return roi_feats, roi_tasks


def train_by_roi_and_task(subjects_partition, tasks, spatial_filters, tensors, scope_name):

    subjects_training, subjects_validation, subjects_test = subjects_partition
    n_filters_  = n_filters
    for task in tasks:
        task_idx = TASKS[task]
        betas_task = np.zeros([NUM_FEATURES+n_additional_features+1, n_filters_])
        task_values = all_tasks_normalized[task_idx:task_idx + 1, :, :]
        learned_weights = {}
        for j in range(n_filters_):
            ind = spatial_filters[: STANDART_BM.N_CORTEX, j] > 0
            print("do regression for task {} filter {} with {} vertices".format(task.name, j, np.size(np.nonzero(ind))))
            if np.size(np.nonzero(ind))<30:
                continue
            roi_feats, roi_tasks = get_selected_features_and_tasks(all_features_normalized, ind, task_values, subjects_training)
            roi_feats_val, roi_tasks_val = get_selected_features_and_tasks(all_features_normalized, ind, task_values, subjects_validation)
            learned_betas = np.linalg.pinv(add_ones_column(roi_feats)).dot(roi_tasks.transpose())
            training = (roi_feats, roi_tasks.transpose())
            validation = (roi_feats_val, roi_tasks_val.transpose())

            print("train : M = {0:.3f}, SD = {1:.3f}".format( np.mean(training[1]), np.std(training[1])))
            # print("val: M = {0:.3f}, SD = {1:.3f}".format(np.mean(validation[1]), np.std(validation[1])))

            strt = time.time()
            _, weights = train_model(tensors, loss_function, training, validation, max_epochs=MAX_EPOCHS_PER_ROI,
                                     batch_size=BATCH_SIZE_R0I,
                                     scope_name=scope_name)
            stp = time.time()
            print("time for training roi {0} = {1:.3f}".format(j, stp-strt))
            predicted = np.dot(add_ones_column(roi_feats), learned_betas)
            predicted_val = np.dot(add_ones_column(roi_feats_val), learned_betas)
            loss = rms_loss(predicted, roi_tasks.transpose())
            loss_val = rms_loss(predicted_val, roi_tasks_val.transpose())
            r_train = np.corrcoef(np.squeeze(predicted), np.squeeze(training[1]))[0, 1]
            r_val = np.corrcoef(np.squeeze(predicted_val), np.squeeze(validation[1]))[0, 1]

            predicted_val_tf = predict_from_model(tensors, roi_feats_val, weights, scope_name)
            predicted_train_tf = predict_from_model(tensors, roi_feats, weights, scope_name)
            loss_tf_train = rms_loss(predicted_train_tf, training[1])
            loss_tf_val = rms_loss(predicted_val_tf, validation[1])
            r_tf_train = np.corrcoef(np.squeeze(predicted_train_tf), np.squeeze(training[1]))[0, 1]
            r_tf_val = np.corrcoef(np.squeeze(predicted_val_tf), np.squeeze(validation[1]))[0, 1]

            # print("training loss pinv = {0:.3f}".format(loss))
            # print("training loss nn = {0:.3f}".format(loss_tf_train))
            # print("validation loss pinv = {0:.3f}".format(loss_val))
            # print("validation loss nn = {0:.3f}".format(loss_tf_val))
            print("training corr pinv = {0:.3f}".format(r_train))
            print("training corr nn = {0:.3f}".format(r_tf_train))
            print("validation corr pinv = {0:.3f}".format(r_val))
            print("validation corr nn = {0:.3f}".format(r_tf_val))

            learned_weights[j] = weights
            betas_task[:, j] = np.squeeze(learned_betas)


        pickle.dump(learned_weights, safe_open(os.path.join(NN_WEIGHTS_PATH, "nn_2hl_no_roi_normalization_fsf_70s_weights_task{0}_all_filters.pkl".
                                                            format(task)), 'wb'))

        pickle.dump(betas_task, safe_open(os.path.join(LINEAR_BETAS_PATH,  "linear_weights_70_s_2", "linear_weights_fsf_task{0}.pkl".
                                                   format(task.name)), 'wb'))
        print("saved betas {}".format(task.name))
    return


def train_linear_original(subjects_partition, spatial_filters):

    subjects_training, subjects_validation, subjects_test = subjects_partition
    n_filters_ = np.size(spatial_filters, axis=1)
    betas = np.zeros([NUM_FEATURES+1, n_filters_, len(TASKS)])
    task_values = all_tasks_normalized[:, :, :]
    learned_weights = {}
    for j in range(n_filters_):
        ind = spatial_filters[: STANDART_BM.N_CORTEX, j] > 0
        print("do regression for filter {} with {} vertices".format(j, np.size(np.nonzero(ind))))
        if np.size(np.nonzero(ind))<30:
            continue
        for s in subjects_training:
            roi_feats, roi_tasks = get_selected_features_and_tasks(all_features_normalized, ind, task_values, [s])
            learned_betas = np.linalg.pinv(add_ones_column(roi_feats)).dot(roi_tasks.transpose())
            betas[:, j, :]+= learned_betas

    betas /= len(subjects_training)
    for task, task_idx in TASKS.items():
        task_betas = betas[:,:,task_idx]
        pickle.dump(task_betas, safe_open(os.path.join(LINEAR_BETAS_PATH,  "linear_weights_70_s_averaged_betas",
                                                  "linear_weights_task{0}.pkl".format(task.name)), 'wb'))
    return



def get_stats(predicted, actual):
    assert np.shape(predicted) == np.shape(actual)
    loss = rms_loss(predicted, actual)
    r = np.corrcoef(predicted, actual)[0, 1]
    return loss, r


def predict_all_subject(subjects, tasks, features_getter, predictor):
    tasks = np.swapaxes(tasks,1,2)
    tasks = demean_and_normalize(tasks[:,:STANDART_BM.N_CORTEX,:], axis=1) # TODO
    #pred = np.empty([tasks.shape[0], STANDART_BM.N_CORTEX , n_subjects])
    pred = {}
    for i, subject in enumerate(subjects):
        print("get features for subject {}".format(subject.subject_id))
        arr = features_getter(subject)
        print("calculate prediction for subject {}".format(subject.subject_id))
        start = time.time()
        pred[:,:,i] = predictor(arr)
        end = time.time()
        print("prediction time: {:.3f}".format(end-start))


def get_correlations_for_subjects(subjects, tasks, features_getter, predictor, tasks_predicted, saved_pred =  None):

    tasks = np.swapaxes(tasks,1,2)
    tasks = demean_and_normalize(tasks[:,:STANDART_BM.N_CORTEX,:], axis=1) # TODO
    n_tasks = np.size(tasks, axis=0)
    n_subjects = len(subjects)
    pred = np.empty([tasks.shape[0], STANDART_BM.N_CORTEX , n_subjects]) if saved_pred is None else saved_pred
    correlations = np.zeros([n_tasks,n_subjects])
    correlations_with_mean = np.zeros([n_tasks, n_subjects])
    correlations_of_mean_roi = np.zeros([n_tasks,n_subjects])
    regions_sizes = [np.count_nonzero(hard_filters[: STANDART_BM.N_CORTEX , j] > 0) for j in range(n_filters)]
    correlations_by_roi = np.zeros([n_tasks,n_subjects, n_filters])
    losses = np.zeros([n_tasks,n_subjects])
    losses2 = np.zeros([n_tasks, n_subjects])
    all_arranged_in_dicts = {}



    if saved_pred is None:
        for i, subject in enumerate(subjects):
            subject_idx = int(subject.subject_id)-1
            print("calculate prediction for subject {}".format(subject.subject_id))
            start = time.time()
            arr = features_getter(subject)
            s_prediction = predictor(arr)
            for task_name, s_task_pred in s_prediction.items():
                pred[TASKS[task_name],:,i] = s_task_pred
            end = time.time()
            print("prediction time: {:.3f}".format(end - start))

    mean_actuals = {}
    for i, subject in enumerate(subjects):
        subject_idx = int(subject.subject_id) - 1
        for task_index in [idx for t, idx in TASKS.items() if t in tasks_predicted]:

            if task_index not in mean_actuals:
                mean_actuals[task_index] = np.mean(
                tasks[task_index, :STANDART_BM.N_CORTEX, [int(subject.subject_id) - 1 for s in subjects]], axis=0)


            task_subject_pred = pred[task_index, :, i]
            task_subject_actual = tasks[task_index,:STANDART_BM.N_CORTEX, subject_idx]

            if task_index not in all_arranged_in_dicts:
                all_arranged_in_dicts[task_index] = {}
            only_mean_roi_prediction = np.zeros(np.shape(task_subject_pred))
            only_mean_roi_actual = np.zeros(np.shape(task_subject_actual))
            all_arranged_in_dicts[task_index][subject] = (task_subject_pred, task_subject_actual)
            for j in range(n_filters):
                ind = hard_filters[: STANDART_BM.N_CORTEX, j] > 0
                if np.size(np.nonzero(ind)) < 30:
                    continue
                only_mean_roi_prediction[ind] = np.mean(task_subject_pred[ind])
                only_mean_roi_actual[ind] = np.mean(task_subject_actual[ind])


            correlations_by_roi[task_index, i, :] = get_corr_by_region(hard_filters, task_subject_actual, task_subject_pred)
            # task_subject_pred = task_subject_pred[some_filters > 0]
            # task_subject_actual = task_subject_actual[some_filters > 0]
            # only_mean_roi_prediction = only_mean_roi_prediction[some_filters > 0]
            # only_mean_roi_actual = only_mean_roi_actual[some_filters > 0]

            correlations[task_index, i] = np.corrcoef(task_subject_actual, task_subject_pred)[0, 1]
            correlations_with_mean[task_index, i] = np.corrcoef(mean_actuals[task_index], task_subject_pred)[0, 1]


            correlations_of_mean_roi[task_index, i] = np.corrcoef(only_mean_roi_actual, only_mean_roi_prediction)[0, 1]

            losses[task_index,i] = rms_loss(task_subject_pred, task_subject_actual)
            losses2[task_index, i] = rms_loss(task_subject_pred, task_subject_actual, True, True)




    for task_index in range(np.size(tasks, axis=0)):

        try:

            task_name = [taskname for taskname, idx in TASKS.items() if idx==task_index ][0]
            print("task = {0} : {1}, loss ={2:.4f}, loss2 = {3:.4f} correlation={4:.4f}, avg_corr_with_mean={5:.4f}, corrs_only_mean_roi={6:.4f}".format(
                task_index+1, task_name, np.mean(losses[task_index,:]),np.mean(losses2[task_index,:])  ,
                np.mean(correlations[task_index,:]), np.mean(correlations_with_mean[task_index, :]),
                np.mean(correlations_of_mean_roi[task_index,:])))
            mean_corrs_by_roi = (np.mean(correlations_by_roi[task_index, :, :], axis=0))


            roi_sizes_weighted = regions_sizes / np.sum(regions_sizes)

            predicted_by_subj = {s : maps[0] for s, maps in all_arranged_in_dicts[task_index].items()}
            actual_by_subj = {s: maps[1] for s, maps in all_arranged_in_dicts[task_index].items()}
            # print("by region:")
            # print(mean_corrs_by_roi)
            # print("correlation with means only:")
            # print(np.sum(mean_corrs_by_roi * roi_sizes_weighted))
            # print("by subject:")
            # print(correlations[task_index,:])
            all_corrs = get_predicted_actual_correlations(subjects, task_name, (predicted_by_subj, actual_by_subj))
            all_corrs_normalized = demean_and_normalize(demean_and_normalize(all_corrs, axis=0), axis=1)

            print("mean corrs = : {0:.3f}".format(np.mean(all_corrs)))
            print("mean self corrs = : {0:.3f}".format(np.mean(np.diag(all_corrs))))
            print("mean self corrs normalized = : {0:.3f}".format(np.mean(np.diag(all_corrs_normalized))))
            print("")


        except KeyError as kerr:
            continue


    #np.save(os.path.join(LOO_betas_path,'all_correlations.npy'), all_correlations)


def get_corr_by_region(filters, pred, act):
    results = np.zeros([n_filters])
    for j in range(n_filters):
        ind = filters[: STANDART_BM.N_CORTEX, j] > 0
        if np.size(np.nonzero(ind)) < 30:
            continue
        results[j] = np.corrcoef(pred[ind], act[ind])[0, 1]
    return results


def predict_from_linear_betas(arr, betas):
    subject_feats = add_ones_column(arr)
    return subject_feats.dot(betas)


def predict_by_roi(subject_feats, filters, saved_weights, tasks, prediction_function):
    subject_predictions = {}
    for task in tasks:
        subject_task_prediction = np.zeros([STANDART_BM.N_CORTEX])
        for j in range(np.size(filters, axis=1)):
            if j in saved_weights[task]:
                weights = saved_weights[task][j]
                ind = filters[: STANDART_BM.N_CORTEX, j] > 0
                weighting =  filters[:,j][ind]
                features = subject_feats[ind]
                subject_task_prediction[ind] += weighting * np.squeeze(prediction_function(features, weights))
        subject_predictions[task] = subject_task_prediction
    return subject_predictions




def run_regression():
    training_size = 70
    validation_size = 30
    n_subjects = training_size + validation_size
    print("files loaded, start regression")
    extracted_featuresr_path = r'D:\Projects\PITECA\Data\extracted features'
    subjects = []
    for i in range(1, n_subjects+1):
        id = general_utils.zeropad(i, 6)
        subjects.append(Subject(subject_id= id,
                                features_path= os.path.join(extracted_featuresr_path, id + '_features.dtseries.nii'),
                                features_exist=True))

    #random.shuffle(subjects)
    tasks_for_model = TASKS.keys()
    subjects_training = subjects[:training_size]
    subjects_validation = subjects[training_size:]
    subjects_test = []
    partition = (subjects_training,subjects_validation, subjects_test)

    # train_by_roi_and_task(partition, tasks_for_model, hard_filters, tensors_two_hidden_layer_nn,
    #                       scope_name=nn_scope_name)

    linear_betas_by_task = {}
    for task in tasks_for_model:
        saved_betas_matrix = pickle.load(
            open(os.path.join(LINEAR_BETAS_PATH, "linear_weights_70_s_2", "linear_weights_fsf_task{0}.pkl".
                                                   format(task.name)), 'rb'))
        # saved_betas_matrix = pickle.load(
        #     safe_open(os.path.join(LINEAR_BETAS_PATH, "linear_weights_70_s_averaged_betas",
        #                             "linear_weights_task{0}.pkl".format(task.name)), 'rb'))
        linear_betas_by_task[task] = {j : saved_betas_matrix[:,j] for j in range(np.size(saved_betas_matrix, axis=1))}

    saved_weights_by_task = {}
    for task in tasks_for_model:
        # saved_weights_by_task[task] = pickle.load(
        #     open(os.path.join(NN_WEIGHTS_PATH, "nn_2hl_no_roi_normalization_70s_weights_task{0}_all_filters.pkl".
        #                       format(task)), 'rb'))
        saved_weights_by_task[task] = pickle.load(
            open(os.path.join(NN_WEIGHTS_PATH, "nn_2hl_no_roi_normalization_fsf_70s_weights_task{0}_all_filters.pkl".
                              format(task)), 'rb'))

    get_subject_features = lambda s : get_subject_features_from_matrix(s)
    predict_subject_tasks_linear = lambda arr : predict_by_roi(
        arr, soft_filters, saved_weights=linear_betas_by_task, tasks = tasks_for_model, prediction_function =
        predict_from_linear_betas)
    predict_from_nn_weights = lambda features, weights : \
        predict_from_model(tensors=tensors_two_hidden_layer_nn,
                       features=features,
                       saved_weights=weights,
                       scope_name = nn_scope_name)
    predict_subject_tasks_nn =  lambda arr : predict_by_roi(
        arr, soft_filters, saved_weights=saved_weights_by_task, tasks = tasks_for_model, prediction_function =
        predict_from_nn_weights)

    prediction = np.load(linear_pred_path_fsf)
    # get_correlations_for_subjects(subjects_validation, all_tasks_normalized,
    #                               get_subject_features, predict_subject_tasks_linear)
    get_correlations_for_subjects(subjects_validation, all_tasks,
                                  get_subject_features, predict_subject_tasks_nn,tasks_for_model, prediction)
    return


if __name__ == "__main__":
    run_regression()

