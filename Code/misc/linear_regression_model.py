import numpy as np
from sklearn.preprocessing import normalize
from py_matlab_utils import *
from constants import *
from linalg_utils import *
import os
import matplotlib.pyplot as plt
import cifti

'''
rough translation of Ido's matlab code of the linear model. This is just a POC,
not to be used in PITECA.

'''


LOO_betas_path = os.path.join(PATHS.DATA_DIR, 'regression_model','loo_betas')
os.makedirs(LOO_betas_path, exist_ok=True)
# file names
AllFeatures_File = os.path.join(r'D:\Projects\PITECA\Data',"all_features.npy")
spatial_filters_file = os.path.join(PATHS.DATA_DIR,"spatial_filters.npy")
task_file = os.path.join(PATHS.DATA_DIR,"Task.npy")
spatial_filters_path =  os.path.join(PATHS.DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')



def linear_regression_on_all_features(all_features, filters, task):
    shape = np.shape(all_features)
    assert shape[0] == N_TOTAL_VERTICES
    n_features = shape[2]
    n_filters = np.size(filters, axis=1)
    n_subjects = shape[1]
    assert n_subjects == 100
    for i in range(n_subjects):
        print("do regression for subject {}".format(i))
        subject_feats = np.empty([N_TOTAL_VERTICES, n_features + 1])
        subject_feats[:,1:] = all_features[:,i,:]
        subject_task = task[i,:]
        subject_feats[CORTEX, :] = demean_and_normalize(subject_feats[CORTEX, :])
        subject_feats[SUBCORTEX, :] = demean_and_normalize(subject_feats[SUBCORTEX, :])
        subject_feats[:,:] = normalize(subject_feats, 'l2', axis=0)
        subject_feats[:,0] = 1.0

        betas = np.zeros([n_features+1, n_filters])
        for j in range(n_filters):
            ind = filters[:,j] > 0
            y = subject_task[ind]
            M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
                                demean(subject_feats[ind, 1:])),axis=1)
            betas[:,j] = np.linalg.pinv(M).dot(y)
        np.save(os.path.join(LOO_betas_path,"betas_subject_{}.npy".format(i)),betas)
    return


def predict_from_betas_LOO(all_features, filters, task):
    task = np.transpose(task)
    shape = np.shape(all_features)
    assert shape[0] == STANDART_BM.N_TOTAL_VERTICES
    n_features = shape[2]
    n_filters = np.size(filters, axis=1)
    n_subjects = shape[1]
    pred = np.empty(task.shape)
    pred_restored = np.empty(task.shape)
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



def run_regression():

    filters = np.load(spatial_filters_file)
    spatial_filters_raw, (series, bm) = cifti.read(spatial_filters_path)
    spatial_filters_raw = np.transpose(spatial_filters_raw)
    task = np.load(task_file)
    all_features = np.load(AllFeatures_File)
    print("files loaded, start regression")
    #linear_regression_on_all_features(all_features, filters,task)
    predict_from_betas_LOO(all_features, spatial_filters_raw,  task)
    return



if __name__ == "__main__":
    run_regression()

