import numpy as np
import tensorflow as tf

from sklearn.preprocessing import normalize
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.io_utils import *
from sharedutils.subject import *
from model.models import *
import os
import matplotlib.pyplot as plt
import cifti
import sharedutils.general_utils as general_utils
import definitions
import random
from misc.nn_model import *

'''
rough translation of Ido's matlab code of the linear model. This is just a POC,
not to be used in PITECA.

'''

LOCAL_DATA_DIR = r'D:\Projects\PITECA\Data'

#LOO_betas_path = os.path.join(LOCAL_DATA_DIR, 'model', 'linear','loo_betas_7_tasks')
LOO_betas_path_tf = os.path.join(LOCAL_DATA_DIR, 'model', 'linear','loo_betas_7_tasks_take3')
LOO_betas_path = os.path.join(LOCAL_DATA_DIR, 'model', 'linear','loo_betas_7_tasks_take2')
os.makedirs(LOO_betas_path, exist_ok=True)
# file names
AllFeatures_File = os.path.join(r'D:\Projects\PITECA\Data',"all_features.npy")
spatial_filters_file = os.path.join(LOCAL_DATA_DIR, 'HCP_200', "spatial_filters.npy")
tasks_file = os.path.join(LOCAL_DATA_DIR, 'HCP_200', "moreTasks.npy")
spatial_filters_path = os.path.join(LOCAL_DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')
subjects_features_order = os.path.join(LOCAL_DATA_DIR, 'subjects_features_order.txt')



def linear_regression_on_features(subjects, filters, tasks, all_features):
    if all_features is None:
        all_features = np.load(AllFeatures_File)
    mapping = get_subject_to_feature_index_mapping(subjects_features_order)
    n_filters = np.size(filters, axis=1)
    n_tasks = np.size(tasks, 0)
    n_features = NUM_FEATURES
    fe = FeatureExtractor()
    for subject in subjects:
        print("do regression for subject {}".format(subject.subject_id))
        arr = get_subject_features_from_matrix(subject, all_features, mapping)
        subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX,1]),np.transpose(arr)),axis=1)
        i = int(subject.subject_id) -1
        subject_tasks = tasks[:,i,:]
        subject_feats[:,:] = normalize(subject_feats, 'l2', axis=0)
        subject_feats[:,0] = 1.0
        betas = np.zeros([n_tasks, n_features+1, n_filters])
        for j in range(n_filters):
            ind = filters[STANDART_BM.CORTEX,j] > 0
            if np.size(np.nonzero(ind))<30:
                continue
            y = np.transpose(subject_tasks[:,ind])
            M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
                                demean(subject_feats[ind, 1:])),axis=1)
            betas[:,:,j] = np.transpose(np.linalg.pinv(M).dot(y))
        np.save(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)),betas)
    return

def linear_regression_on_features_tf(subjects, filters, tasks, all_features):
    if all_features is None:
        all_features = np.load(AllFeatures_File)
    mapping = get_subject_to_feature_index_mapping(subjects_features_order)
    n_filters = np.size(filters, axis=1)
    n_tasks = np.size(tasks, 0)
    n_features = NUM_FEATURES
    fe = FeatureExtractor()
    for subject in subjects:
        print("do regression for subject {}".format(subject.subject_id))
        arr = get_subject_features_from_matrix(subject, all_features, mapping)
        subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX,1]),np.transpose(arr)),axis=1)
        i = int(subject.subject_id) -1
        subject_tasks = tasks[:,i,:]
        subject_feats[:,:] = normalize(subject_feats, 'l2', axis=0)
        subject_feats[:,0] = 1.0
        betas = np.zeros([n_tasks, n_features+1, n_filters])
        for j in range(n_filters):
            ind = filters[STANDART_BM.CORTEX,j] > 0
            if np.size(np.nonzero(ind))<30:
                continue
            y = np.transpose(subject_tasks[:,ind])
            M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
                                demean(subject_feats[ind, 1:])),axis=1)
            #betas[:,:,j] = np.transpose(np.linalg.pinv(M).dot(y))
            betas[:, :, j] = get_betas(M, y)
        np.save(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)),betas)
    return


def get_selected_features_and_tasks(all_features, ind, tasks, mapping, subjects):
    roi = [demean_and_normalize(all_features[ind, mapping[int(s.subject_id)], :], axis=0) for s in subjects]
    roi_feats = np.concatenate(roi, axis=0)
    roi_feats = np.concatenate((np.ones([np.size(roi_feats, 0), 1]), roi_feats), axis=1)
    roi_tasks = [tasks[:, int(s.subject_id) - 1, ind] for s in subjects]
    roi_tasks = np.concatenate(roi_tasks, axis=1)
    roi_feats = demean_and_normalize(roi_feats, axis=0)
    roi_feats[:, 0] = 1.0
    return roi_feats, roi_tasks


def linear_regression_on_features_tf_on_all(subjects_training, subjects_validation, filters, all_tasks, all_features=None):
    if all_features is None:
        all_features = np.load(AllFeatures_File)

    all_features = all_features[:STANDART_BM.N_CORTEX,:]
    all_tasks = all_tasks[:,:, :STANDART_BM.N_CORTEX]
    mapping = get_subject_to_feature_index_mapping(subjects_features_order)
    n_filters = np.size(filters, axis=1)
    n_tasks = np.size(all_tasks, 0)
    n_features = NUM_FEATURES
    betas = np.zeros([NUM_FEATURES+1, n_tasks, n_filters])
    filter_sizes = [np.size(np.nonzero( filters[: STANDART_BM.N_CORTEX,j])) for j in range(n_filters)]
    for task_idx in range(n_tasks):
        task = all_tasks[task_idx:task_idx+1,:,:]
        for j in range(n_filters):

            ind = filters[: STANDART_BM.N_CORTEX,j] > 0
            print("do regression for filter {} with {} vertices".format(j, np.size(np.nonzero(ind))))
            if np.size(np.nonzero(ind))<30:
                continue
            roi_feats, roi_tasks = get_selected_features_and_tasks(all_features, ind, task, mapping, subjects_training)
            roi_feats_val, roi_tasks_val = get_selected_features_and_tasks(all_features, ind, task, mapping, subjects_validation)
            learned_betas = np.linalg.pinv(roi_feats).dot(roi_tasks.transpose())
            predicted = np.dot(roi_feats, learned_betas)
            predicted_val = np.dot(roi_feats_val, learned_betas)
            loss = np.mean(np.square(predicted - roi_tasks.transpose()))
            loss_val = np.mean(np.square(predicted_val - roi_tasks_val.transpose()))
            print("training loss pinv = {0:.3f}".format(loss))
            print("validation loss pinv = {0:.3f}".format(loss_val))

            for l in  [i*1e-3 for i in [5]]:
                training = (roi_feats, roi_tasks.transpose())
                validation = (roi_feats_val, roi_tasks_val.transpose())
                loss_tf_val, weights = learn_one_region(
                    training, validation ,max_epochs=1000 ,batch_size=50, reg_lambda=l)
                print("reg lambda = {0:.6f}".format(l))
                print("validation loss tf = {0:.3f}".format(loss_tf_val))


    np.save(os.path.join(LOO_betas_path,"betas_regressed_on_all_tf.npy"),betas)
    return


def get_subject_to_feature_index_mapping(path):
    mapping = {}
    with open(path, 'r') as f:
        for i in range(100):
            subj_number = int(f.readline()) + 1
            assert subj_number not in mapping
            mapping[subj_number] = i
    return mapping

def get_subject_features_from_matrix(subject, all_features, mapping, t=False):
    subj_features = all_features[STANDART_BM.CORTEX, mapping[int(subject.subject_id)], :]
    if t:
        subj_features = subj_features.transpose()
    return subj_features.transpose()


def predict_from_betas_LOO_fromSubjects_fast(subjects, filters, tasks, all_features):
    if all_features is None:
        all_features = np.load(AllFeatures_File)
    tasks = np.swapaxes(tasks,1,2)
    fe = FeatureExtractor()
    subjects_features = {}
    all_correlations = np.empty([np.size(tasks, axis=0), len(subjects)])
    n_tasks = np.size(tasks, axis=0)
    all_features = np.load(AllFeatures_File)
    mapping = get_subject_to_feature_index_mapping(subjects_features_order)


    n_features = NUM_FEATURES
    n_filters = np.size(filters, axis=1)
    n_subjects = len(subjects)
    pred = np.empty(tasks.shape)
    # betas = np.empty([n_tasks, n_features + 1, n_filters, n_subjects])
    # for i in range(n_subjects):
    #     subj_betas = np.load(os.path.join(LOO_betas_path_tf,"betas_subject_{}.npy".format(i)))
    #     betas[:,:,:,i] = subj_betas
    # average_betas = np.mean(betas, axis=3)

    betas = np.load(os.path.join(LOO_betas_path,"betas_regressed_on_all.npy"))
    betas = betas.swapaxes(0,1)
    #np.save(os.path.join(LOO_betas_path, "average_betas_100_subject_all_tasks.npy"), average_betas)
    filters = (filters[STANDART_BM.CORTEX, :]).astype(float)
    pred = pred[:, :STANDART_BM.N_CORTEX,:]
    temperature = 3.5
    loo_correlations = np.zeros([n_tasks,n_subjects])
    correlations = np.zeros(n_subjects)
    self_correlations = np.zeros(n_subjects)
    tempered_filters = softmax(filters* temperature) if temperature != 'even' \
        else filters / np.reshape(np.sum(filters, axis=1), [STANDART_BM.N_CORTEX,1])

    for i, subject in enumerate(subjects):
        print("get features for subject {}".format(subject.subject_id))
        #arr, bm = fe.get_features(subject)
        arr = get_subject_features_from_matrix(subject, all_features, mapping)
        subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX, 1]), np.transpose(arr)), axis=1)
        subject_feats = demean_and_normalize(subject_feats)
        subject_feats[:, 0] = 1.0

        #print("calculate prediction for subject {}".format(subject.subject_id))
        loo_betas = betas #(average_betas * n_subjects - betas[:,:,:,i])/(n_subjects-1)
        dotprod = subject_feats.dot(loo_betas)

        pred[:,:,i] = np.sum(np.swapaxes(dotprod, 0, 1) * tempered_filters, axis=2)
        for task_index in range(np.size(tasks, axis=0)):
            loo_correlations[task_index,i] = \
            np.corrcoef(tasks[task_index,STANDART_BM.CORTEX, i], pred[task_index,:, i])[0, 1]
        # pred[:, i] = np.sum(subject_feats.dot(average_betas) * tempered_filters, axis=1)
        # correlations[i] = \
        # np.corrcoef(task[STANDART_BM.CORTEX, i], pred[:, i])[0, 1]
        # pred[:, i] = np.sum(subject_feats.dot(betas[:,:,i]) * tempered_filters, axis=1)
        # self_correlations[i] = \
        # np.corrcoef(task[STANDART_BM.CORTEX, i], pred[:, i])[0, 1]

    #all_correlations[task_index, :] = correlations
    for task_index in range(np.size(tasks, axis=0)):
        print("task = {0}, temperature = {1}, loo_correlation={2:.4f}".format(
            task_index+1,temperature, np.mean(loo_correlations[task_index,:])))
    #np.save(os.path.join(LOO_betas_path,'all_correlations.npy'), all_correlations)


def basic_linear_regression_build(n_features, n_tasks):
    x = tf.placeholder(tf.float32, shape=(None, n_features), name='x')
    y = tf.placeholder(tf.float32, shape=(None, n_tasks), name='y')

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(tf.random_normal([n_features,n_tasks]), name = 'w')
        y_pred = tf.matmul(x,w)
        loss = tf.reduce_mean(tf.square(y_pred-y))
    return x, y, y_pred, loss
#
# def regression_with_one_hidden_leyer_build(n_features, n_tasks, reg_lambda=0):
#     x = tf.placeholder(tf.float32, shape=(None, n_features), name='x')
#     y = tf.placeholder(tf.float32, shape=(None, n_tasks), name='y')
#
#     with tf.variable_scope('nlreg') as scope:
#         w1 = tf.Variable(tf.random_normal([n_features, n_features * 2]), name ='w1')
#         b1 = tf.Variable(tf.random_normal([n_features *2]), name ='b1')
#         w2 = tf.Variable(tf.random_normal([n_features *  2, n_tasks]), name ='w2')
#         b2 = tf.Variable(tf.random_normal([n_tasks]), name = 'b2')
#
#         hidden_layer = tf.nn.relu(tf.matmul(x, w1) + b1)
#         y_pred = tf.matmul(hidden_layer, w2 + b2)
#         regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
#         loss = tf.reduce_mean(tf.square(y_pred-y)) + reg_lambda * regularizer
#     return x, y, y_pred, loss
#
#
# def learn_one_region(features, activation, features_val, n_epochs, batch_size, reg_lambda = 0):
#     check_every = min(2 *int(np.size(features, 0) // batch_size), 5000)
#     dataset = Dataset(features, activation)
#     n_samples = np.size(features, 0)
#     x, y, y_pred, loss = regression_with_one_hidden_leyer_build(np.size(features, 1), np.size(activation, 1),
#                                                                 reg_lambda = reg_lambda)
#     optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)
#     init = tf.global_variables_initializer()
#     losses = np.zeros([check_every])
#     current_avg_loss = float('inf')
#     curr_loss = float('inf')
#     with tf.Session() as session:
#         session.run(init)
#         idx = 0
#         while dataset.epochs_completed < n_epochs:
#
#             x_batch, y_batch = dataset.next_batch(batch_size= batch_size)
#             feed_dict = {x: x_batch, y: y_batch}
#             losses[idx], _ = session.run([loss, optimizer], feed_dict=feed_dict)
#             idx +=1
#             if idx ==check_every:
#                 idx=0
#                 next_loss = session.run(loss, feed_dict={x: features, y: activation})
#                 next_avg_loss= np.mean(losses)
#                 #print("curr_loss = {0:.2f}, avg_loss on last {1} batches = {2:.2f}".format(curr_loss, check_every,next_avg_loss))
#                 if next_avg_loss < current_avg_loss:
#                     current_avg_loss = next_avg_loss
#                 if next_loss < curr_loss:
#                     curr_loss = next_loss
#
#
#                 #epoch_loss = np.mean(losses[i - 200:i])
#                 #print(epoch_loss)
#                 #if np.mean(losses[i-100:i-50]) - np.mean(losses[i-50:i]) < 0.01:
#                  #   break
#
#         #w1, b1, w2, b2 = session.run(tf.trainable_variables())
#         prediction = session.run(y_pred, feed_dict={x: features})
#         prediction_val = session.run(y_pred, feed_dict={x: features_val})
#
#         return prediction, prediction_val
#
#

def run_regression():
    training_size = 70
    validation_size = 30
    n_subjects = training_size + validation_size
    filters = np.load(spatial_filters_file)
    spatial_filters_raw, (series, bm) = cifti.read(spatial_filters_path)
    spatial_filters_raw = np.transpose(spatial_filters_raw)
    tasks = np.load(tasks_file)
    print("files loaded, start regression")
    extracted_featuresr_path = r'D:\Projects\PITECA\Data\extracted features'
    subjects = []
    for i in range(1, n_subjects+1):
        id = general_utils.zeropad(i, 6)
        subjects.append(Subject(subject_id= id,
                                features_path= os.path.join(extracted_featuresr_path, id + '_features.dtseries.nii'),
                                features_exist=True))

    random.shuffle(subjects)
    subjects_training = subjects[:training_size]
    subjects_validation = subjects[training_size:]
    linear_regression_on_features_tf_on_all(subjects_training, subjects_validation, filters, tasks)
    #predict_from_betas_LOO_fromSubjects_fast(subjects, spatial_filters_raw,  tasks, all_features)
    return



if __name__ == "__main__":
    run_regression()

