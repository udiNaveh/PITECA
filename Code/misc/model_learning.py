import numpy as np
import tensorflow as tf


from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.io_utils import *
from sharedutils.subject import *
from sharedutils.general_utils import safe_open
from model.models import *
import os
import matplotlib.pyplot as plt
import cifti
import sharedutils.general_utils as general_utils
import definitions
import random
import pickle
from misc.nn_model import *

LOO_betas_path_tf = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'linear','loo_betas_7_tasks_take3')
LOO_betas_path = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'linear','loo_betas_7_tasks_take2')
nn_weights_path = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'nn')
# file names

spatial_filters_file = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', "spatial_filters.npy")
tasks_file = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', "moreTasks.npy")
spatial_filters_path = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')
subjects_features_order = os.path.join(definitions.LOCAL_DATA_DIR, 'subjects_features_order.txt')
all_features_path = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order.npy")


all_features = np.load(all_features_path)


def get_subject_features_from_matrix(subject, all_features):
    subj_features = all_features[: STANDART_BM.N_CORTEX, int(subject.subject_id) -1, :]
    return subj_features.transpose()


def get_selected_features_and_tasks(all_features, ind, tasks, subjects):
    roi = [demean_and_normalize(all_features[ind, int(s.subject_id)-1, :], axis=0) for s in subjects]
    roi_feats = np.concatenate(roi, axis=0)
    roi_tasks = [tasks[:, int(s.subject_id) - 1, ind] for s in subjects]
    roi_tasks = np.concatenate(roi_tasks, axis=1)
    return roi_feats, roi_tasks



def lin_nonlin_reg_all_features(subjects_training, subjects_validation, filters, all_tasks):

    all_tasks = all_tasks[:,:, :STANDART_BM.N_CORTEX]
    mapping = get_subject_to_feature_index_mapping(subjects_features_order)
    n_filters = np.size(filters, axis=1)
    n_tasks = np.size(all_tasks, 0)
    n_features = NUM_FEATURES
    betas = np.zeros([NUM_FEATURES+1, n_tasks, n_filters])
    filter_sizes = [np.size(np.nonzero( filters[: STANDART_BM.N_CORTEX,j])) for j in range(n_filters)]

    tensors = regression_with_one_hidden_layer_build(NUM_FEATURES, 1)
    best_weights = {}
    for task_idx in range(n_tasks):
        betas_task = np.zeros([NUM_FEATURES+1, n_filters])
        pickle.dump(betas_task, safe_open(os.path.join(LOO_betas_path, "linear_weights_70_s","linear_weights_task{0}.pkl".
                                               format(task_idx)), 'wb'))
        task = all_tasks[task_idx:task_idx+1,:,:]
        task = demean_and_normalize(task, axis=2)
        for j in range(n_filters):
            ind = filters[: STANDART_BM.N_CORTEX,j] > 0
            print("do regression for filter {} with {} vertices".format(j, np.size(np.nonzero(ind))))
            if np.size(np.nonzero(ind))<30:
                continue
            roi_feats, roi_tasks = get_selected_features_and_tasks(all_features, ind, task, subjects_training,)
            roi_feats_val, roi_tasks_val = get_selected_features_and_tasks(all_features, ind, task, subjects_validation)
            learned_betas = np.linalg.pinv(add_ones_column(roi_feats)).dot(roi_tasks.transpose())
            training = (roi_feats, roi_tasks.transpose())
            validation = (roi_feats_val, roi_tasks_val.transpose())

            predicted = np.dot(add_ones_column(roi_feats), learned_betas)
            predicted_val = np.dot(add_ones_column(roi_feats_val), learned_betas)

            loss = rms_loss(predicted, roi_tasks.transpose())
            loss_val = rms_loss(predicted_val, roi_tasks_val.transpose())


            l = 0.03

            _, weights = train_model(
                tensors, training, validation ,max_epochs=200 ,batch_size=200, regularization_lambda=l)
            predicted_val_tf = predict_from_model(tensors, roi_feats_val, weights)
            predicted_train_tf = predict_from_model(tensors, roi_feats, weights)
            loss_tf_train = rms_loss(predicted_train_tf, training[1])
            loss_tf_val = rms_loss(predicted_val_tf, validation[1])
            best_weights[j] = weights
            print("training loss pinv = {0:.3f}".format(loss))
            print("validation loss pinv = {0:.3f}".format(loss_val))
            print("training loss nn = {0:.3f}".format(loss_tf_train))
            print("validation loss2 nn = {0:.3f}".format(loss_tf_val))

            pickle.dump(weights, open(os.path.join(nn_weights_path,"nn_2hl_weights_task{0}_filter{1}.pkl".
                                                   format(task_idx, j)),'wb'))
            betas_task[:, j] = np.squeeze(learned_betas)

        pickle.dump(betas_task, safe_open(os.path.join(LOO_betas_path, "linear_weights_70_s","linear_weights_task{0}.pkl".
                                               format(task_idx)), 'wb'))


    return


def get_correlations_for_subjects(subjects, tasks, features_getter, predictor):

    tasks = np.swapaxes(tasks,1,2)
    n_tasks = np.size(tasks, axis=0)
    n_subjects = len(subjects)
    pred = np.empty([tasks.shape[0], STANDART_BM.N_CORTEX , n_subjects])
    correlations = np.zeros([n_tasks,n_subjects])
    losses = np.zeros([n_tasks,n_subjects])
    losses2 = np.zeros([n_tasks, n_subjects])


    for i, subject in enumerate(subjects):
        subject_idx = int(subject.subject_id)-1
        print("get features for subject {}".format(subject.subject_id))
        arr = features_getter(subject)
        print("calculate prediction for subject {}".format(subject.subject_id))
        start = time.time()
        pred[:,:,i] = predictor(arr)
        end = time.time()
        print("prediction time: {:.3f}".format(end-start))
        for task_index in range(np.size(tasks, axis=0)):
            task_subject_pred = pred[task_index,:, i]
            task_subject_actual = tasks[task_index,STANDART_BM.CORTEX, subject_idx]
            correlations[task_index,i] = np.corrcoef(task_subject_actual,task_subject_pred)[0, 1]
            losses[task_index,i] = rms_loss(tasks[task_index,STANDART_BM.CORTEX, subject_idx], pred[task_index,:, i])
            losses2[task_index, i] = rms_loss(tasks[task_index, STANDART_BM.CORTEX, subject_idx], pred[task_index, :, i], True, True)


    for task_index in range(np.size(tasks, axis=0)):
        print("task = {0}, loss ={1:.4f}, loss2 = {2:.4f} correlation={3:.4f}".format(
            task_index+1, np.mean(losses[task_index,:]),np.mean(losses2[task_index,:])  ,
            np.mean(correlations[task_index,:])))
    #np.save(os.path.join(LOO_betas_path,'all_correlations.npy'), all_correlations)


def predict_from_linear_betas(arr, tempered_filters, betas):
    subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX, 1]), np.transpose(arr)), axis=1)
    subject_feats = demean_and_normalize(subject_feats)
    subject_feats[:, 0] = 1.0

    dotprod = subject_feats.dot(betas)
    subject_prediction = np.sum(np.swapaxes(dotprod, 0, 1) * tempered_filters, axis=2)
    return subject_prediction


def predict_from_nn_model(arr, filters, saved_weights, n_features, n_tasks, tensors):
    subject_feats =  np.transpose(arr)
    # subject_feats = demean_and_normalize(subject_feats)
    # subject_feats[:, 0] = 1.0
    subject_prediction = np.zeros([n_tasks, STANDART_BM.N_CORTEX])
    for task_index in range(n_tasks):

        for j in range(n_features):
            ind = filters[: STANDART_BM.N_CORTEX, j] > 0
            if np.size(np.nonzero(ind))<30:
                continue
            features = demean_and_normalize(subject_feats[ind])
            if (task_index, j) not in saved_weights:
                weights = pickle.load(open(os.path.join(nn_weights_path,"nn_2hl_weights_task{0}_filter{1}.pkl".
                                                   format(task_index, j)),'rb'))
                saved_weights[(task_index, j)] = weights
            weights = saved_weights[(task_index, j)]
            subject_prediction[task_index, ind] = np.squeeze(predict_from_model(tensors, features, weights))
    return subject_prediction


def run_regression():
    training_size = 70
    validation_size = 30
    n_subjects = training_size + validation_size
    filters = np.load(spatial_filters_file)
    spatial_filters_raw, (series, bm) = cifti.read(spatial_filters_path)
    spatial_filters_raw = np.transpose(spatial_filters_raw)
    spatial_filters_hard = np.argmax(spatial_filters_raw[:, :STANDART_BM.N_CORTEX], axis=1)


    tasks = np.load(tasks_file)
    tasks = tasks[:1,:,:] # only the first
    print("files loaded, start regression")
    extracted_featuresr_path = r'D:\Projects\PITECA\Data\extracted features'
    subjects = []
    for i in range(1, n_subjects+1):
        id = general_utils.zeropad(i, 6)
        subjects.append(Subject(subject_id= id,
                                features_path= os.path.join(extracted_featuresr_path, id + '_features.dtseries.nii'),
                                features_exist=True))

    #random.shuffle(subjects)

    subjects_training = subjects[:training_size]
    subjects_validation = subjects[training_size:]

    #lin_nonlin_reg_all_features(subjects_training, subjects_validation, filters, tasks, all_features)

    # betas = np.load(os.path.join(LOO_betas_path,"betas_regressed_on_all.npy"))
    # betas = betas.swapaxes(0,1)
    # print(betas.shape)
    betas_by_task = []
    for task_idx in range(1):
        betas_task = pickle.load(open(os.path.join(LOO_betas_path, "linear_weights_70_s","linear_weights_task{0}.pkl".
                                                  format(task_idx)), 'rb'))
        betas_by_task.append(betas_task)
    betas_by_task = np.stack(betas_by_task, axis=0)

    tempered_filters = {}
    temperature = 3.5
    tempered_filters = softmax((spatial_filters_raw[STANDART_BM.CORTEX, :]).astype(float)  * temperature)
    get_subject_features = lambda s : get_subject_features_from_matrix(s, all_features)
    predict_subject_tasks_linear = lambda arr : predict_from_linear_betas(arr, tempered_filters, betas_by_task)


    tensors = regression_with_one_hidden_layer_build(NUM_FEATURES, 1)
    saved_weights = {}
    predict_subject_tasks_nn = lambda arr:\
        predict_from_nn_model(arr, filters, saved_weights, 50, 1, tensors)
    get_correlations_for_subjects(subjects_validation, tasks
                                  , get_subject_features, predict_subject_tasks_linear)
    get_correlations_for_subjects(subjects_validation, tasks
                                  , get_subject_features, predict_subject_tasks_nn)
    return



if __name__ == "__main__":
    run_regression()

