import numpy as np
from sklearn.preprocessing import normalize
from sharedutils.constants import *
from sharedutils.linalg_utils import *
from sharedutils.io_utils import *
from sharedutils.subject import *
from model.models import *
import os
import matplotlib.pyplot as plt
import cifti
import sharedutils.string_utils as string_utils

'''
rough translation of Ido's matlab code of the linear model. This is just a POC,
not to be used in PITECA.

'''


LOO_betas_path = os.path.join(PATHS.DATA_DIR, 'regression_model','loo_betas_7_tasks')
os.makedirs(LOO_betas_path, exist_ok=True)
# file names
AllFeatures_File = os.path.join(r'D:\Projects\PITECA\Data',"all_features.npy")
spatial_filters_file = os.path.join(PATHS.DATA_DIR,"spatial_filters.npy")
task_file = os.path.join(PATHS.DATA_DIR,"Task.npy")
tasks_file = os.path.join(PATHS.DATA_DIR,"moreTasks.npy")
spatial_filters_path = os.path.join(PATHS.DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')


def linear_regression_on_all_features(all_features, filters, tasks):
    shape = np.shape(all_features)
    assert shape[0] == STANDART_BM.N_TOTAL_VERTICES
    n_features = shape[2]
    n_filters = np.size(filters, axis=1)
    n_subjects = shape[1]
    n_tasks = np.size(tasks, 0)
    assert n_subjects == 100
    for i in range(n_subjects):
        print("do regression for subject {}".format(i))
        subject_feats = np.empty([STANDART_BM.N_TOTAL_VERTICES, n_features + 1])
        subject_feats[:,1:] = all_features[:,i,:]
        subject_tasks = tasks[:,i,:]
        subject_feats[STANDART_BM.CORTEX, :] = demean_and_normalize(subject_feats[STANDART_BM.CORTEX, :])
        subject_feats[STANDART_BM.SUBCORTEX, :] = demean_and_normalize(subject_feats[STANDART_BM.SUBCORTEX, :])
        subject_feats[:,:] = normalize(subject_feats, 'l2', axis=0)
        subject_feats[:,0] = 1.0

        betas = np.zeros([n_tasks, n_features+1, n_filters])
        for j in range(n_filters):
            ind = filters[:,j] > 0
            y = np.transpose(subject_tasks[:,ind])
            M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
                                demean(subject_feats[ind, 1:])),axis=1)
            betas[:,:,j] = np.transpose(np.linalg.pinv(M).dot(y))
        np.save(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)),betas)
    return


def linear_regression_on_features(subjects, filters, tasks):


    n_filters = np.size(filters, axis=1)
    n_tasks = np.size(tasks, 0)
    n_features = NUM_FEATURES
    fe = FeatureExtractor()
    for subject in subjects:
        print("do regression for subject {}".format(subject.subject_id))

        arr, bm = fe.get_features(subject)
        subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX,1]),np.transpose(arr)),axis=1)
        i = int(subject.subject_id) -1
        subject_tasks = tasks[:,i,:]
        #subject_feats[STANDART_BM.CORTEX, :] = demean_and_normalize(subject_feats[STANDART_BM.CORTEX, :])
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
        #np.save(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)),betas)
    return


def predict_from_betas_LOO_allFeaturesMatrix(all_features, filters, task):
    task = np.transpose(task)
    shape = np.shape(all_features)
    assert shape[0] == STANDART_BM.N_TOTAL_VERTICES
    n_features = shape[2]
    n_filters = np.size(filters, axis=1)
    n_subjects = shape[1]
    pred = np.empty(task.shape)
    betas = np.empty([n_features + 1, n_filters, n_subjects])
    for i in range(n_subjects):
        betas[:,:,i] = np.load(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)))
    average_betas = np.mean(betas, axis=2)
    np.save(os.path.join(LOO_betas_path, "average_betas_100_subject.npy".format(i)), average_betas)
    filters = (filters[STANDART_BM.CORTEX, :]).astype(float)
    orphans_vertices= np.sum(filters, axis=1) == 0
    filters[orphans_vertices, :] = 1/n_filters
    pred = pred[STANDART_BM.CORTEX,:]
    for temperature in [3.5, 100000]:
        tempered_filters = softmax(filters* temperature)
        for i in range(n_subjects):

            subject_feats = np.empty([STANDART_BM.N_TOTAL_VERTICES, n_features + 1])
            subject_feats[:,1:] = all_features[:,i,:]
            subject_feats= demean_and_normalize(subject_feats[STANDART_BM.CORTEX, :])
            subject_feats[:,0] = 1.0
            loo_betas = (average_betas * n_subjects - betas[:,:,i])/(n_subjects-1)
            pred[:,i] = np.sum(subject_feats.dot(loo_betas) * tempered_filters, axis=1)
            # for j in range(n_filters):
            #     ind = filters[:,j] > 0
            #     M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
            #                         demean(subject_feats[ind, 1:])),axis=1)
            #     M = subject_feats[ind, :]
            #     pred[ind,i] = M.dot(loo_betas[:,j])

            # the above code is commented out as I replaced it with one line of code,
            # with only matrix operations. this will make it easy to implement in tf.


        #np.save(os.path.join(LOO_betas_path,"LOO_predictions_for_all_subjects.npy"), pred)

        correlations = np.zeros(n_subjects)

        for i in range(n_subjects):
            #correlations[i] = np.corrcoef(task[np.bitwise_not(orphans_vertices), i], pred[np.bitwise_not(orphans_vertices), i])[0, 1]
            correlations[i] = \
            np.corrcoef(task[STANDART_BM.CORTEX, i], pred[STANDART_BM.CORTEX, i])[0, 1]

        #correlation_matrix = np.corrcoef(task[np.sum(filters, axis=1) == 1, :], pred[np.sum(filters, axis=1) == 1, :], rowvar=0)[:n_subjects, n_subjects:]
        correlation_matrix = np.corrcoef(task[STANDART_BM.CORTEX, :], pred[STANDART_BM.CORTEX, :], rowvar=0)[:n_subjects, n_subjects:]


        #np.save(os.path.join(DATA_DIR, 'regression_model',"correlation_array.npy"), correlations)
        #np.save(os.path.join(DATA_DIR, 'regression_model',"correlation_matrix.npy"), correlation_matrix)
        # print(correlations)
        # plt.imshow(correlation_matrix, cmap='summer', interpolation='nearest')
        # plt.show()
        print(temperature, np.mean(correlations))



def predict_from_betas_LOO_fromSubjects(subjects, filters, tasks):

    fe = FeatureExtractor()
    subjects_features = {}
    all_correlations = np.empty([np.size(tasks, axis=0), len(subjects)])


    for task_index in range(np.size(tasks, axis=0)):
        task = np.transpose(tasks[task_index,:,:])
        n_features = NUM_FEATURES
        n_filters = np.size(filters, axis=1)
        n_subjects = len(subjects)
        pred = np.empty(task.shape)
        betas = np.empty([n_features + 1, n_filters, n_subjects])
        for i in range(n_subjects):
            subj_betas = np.load(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)))
            betas[:,:,i] = subj_betas[task_index,:,:]
        average_betas = np.mean(betas, axis=2)
        np.save(os.path.join(LOO_betas_path, "average_betas_100_subject_task_{}.npy".format(task_index)), average_betas)
        filters = (filters[STANDART_BM.CORTEX, :]).astype(float)
        pred = pred[STANDART_BM.CORTEX,:]
        for temperature in ['even',0, 3.5, 100000]:
            loo_correlations = np.zeros(n_subjects)
            correlations = np.zeros(n_subjects)
            self_correlations = np.zeros(n_subjects)
            tempered_filters = softmax(filters* temperature) if temperature != 'even' \
                else filters / np.reshape(np.sum(filters, axis=1), [STANDART_BM.N_CORTEX,1])

            for i, subject in enumerate(subjects):
                if subject not in subjects_features:
                    print("get features for subject {}".format(subject.subject_id))
                    arr, bm = fe.get_features(subject)
                    subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX, 1]), np.transpose(arr)), axis=1)
                    subject_feats = demean_and_normalize(subject_feats)
                    subject_feats[:, 0] = 1.0
                    subjects_features[subject] = subject_feats

                #print("calculate prediction for subject {}".format(subject.subject_id))
                subject_feats = subjects_features[subject]
                loo_betas = (average_betas * n_subjects - betas[:,:,i])/(n_subjects-1)
                pred[:,i] = np.sum(subject_feats.dot(loo_betas) * tempered_filters, axis=1)
                loo_correlations[i] = \
                np.corrcoef(task[STANDART_BM.CORTEX, i], pred[:, i])[0, 1]
                pred[:, i] = np.sum(subject_feats.dot(average_betas) * tempered_filters, axis=1)
                correlations[i] = \
                np.corrcoef(task[STANDART_BM.CORTEX, i], pred[:, i])[0, 1]
                pred[:, i] = np.sum(subject_feats.dot(betas[:,:,i]) * tempered_filters, axis=1)
                self_correlations[i] = \
                np.corrcoef(task[STANDART_BM.CORTEX, i], pred[:, i])[0, 1]

            all_correlations[task_index, :] = correlations

            print("task = {0}, temperature = {1}, mean correlation = {2:.4f}, loo_correlation={3:.4f}, self_correlation={4:.4f}".format(
                task_index+1,temperature, np.mean(correlations), np.mean(loo_correlations), np.mean(self_correlations)))
    np.save(os.path.join(LOO_betas_path,'all_correlations.npy'), all_correlations)

def predict_from_betas_LOO_fromSubjects_fast(subjects, filters, tasks):
    tasks = np.swapaxes(tasks,1,2)
    fe = FeatureExtractor()
    subjects_features = {}
    all_correlations = np.empty([np.size(tasks, axis=0), len(subjects)])
    n_tasks = np.size(tasks, axis=0)


    #for task_index in range(np.size(tasks, axis=0)):
        #task = np.transpose(tasks[task_index,:,:])
    n_features = NUM_FEATURES
    n_filters = np.size(filters, axis=1)
    n_subjects = len(subjects)
    pred = np.empty(tasks.shape)
    betas = np.empty([n_tasks, n_features + 1, n_filters, n_subjects])
    for i in range(n_subjects):
        subj_betas = np.load(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)))
        betas[:,:,:,i] = subj_betas
    average_betas = np.mean(betas, axis=3)
    np.save(os.path.join(LOO_betas_path, "average_betas_100_subject_all_tasks.npy"), average_betas)
    filters = (filters[STANDART_BM.CORTEX, :]).astype(float)
    pred = pred[:, :STANDART_BM.N_CORTEX,:]
    for temperature in ['even',0, 3.5, 100000]:
        loo_correlations = np.zeros([n_tasks,n_subjects])
        correlations = np.zeros(n_subjects)
        self_correlations = np.zeros(n_subjects)
        tempered_filters = softmax(filters* temperature) if temperature != 'even' \
            else filters / np.reshape(np.sum(filters, axis=1), [STANDART_BM.N_CORTEX,1])

        for i, subject in enumerate(subjects):
            if subject not in subjects_features:
                print("get features for subject {}".format(subject.subject_id))
                arr, bm = fe.get_features(subject)
                subject_feats = np.concatenate((np.ones([STANDART_BM.N_CORTEX, 1]), np.transpose(arr)), axis=1)
                subject_feats = demean_and_normalize(subject_feats)
                subject_feats[:, 0] = 1.0
                subjects_features[subject] = subject_feats

            #print("calculate prediction for subject {}".format(subject.subject_id))
            subject_feats = subjects_features[subject]
            loo_betas = (average_betas * n_subjects - betas[:,:,:,i])/(n_subjects-1)
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


def run_regression():

    filters = np.load(spatial_filters_file)
    spatial_filters_raw, (series, bm) = cifti.read(spatial_filters_path)
    spatial_filters_raw = np.transpose(spatial_filters_raw)
    tasks = np.load(tasks_file)
    #all_features = np.load(AllFeatures_File)
    print("files loaded, start regression")
    extracted_featuresr_path = r'D:\Projects\PITECA\Data\extracted features'
    subjects = []
    for i in range(1, 101):
        id = string_utils.zeropad(i, 6)
        subjects.append(Subject(subject_id= id,
                                features_path= os.path.join(extracted_featuresr_path, id + '_features.dtseries.nii'),
                                features_exist=True))
    #linear_regression_on_features(subjects, filters, tasks)
    predict_from_betas_LOO_fromSubjects_fast(subjects, spatial_filters_raw,  tasks)
    return




if __name__ == "__main__":

    run_regression()


