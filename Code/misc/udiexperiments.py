from sharedutils.io_utils import *
from sharedutils.subject import *
from sharedutils.linalg_utils import *
import time
import os

import definitions

from misc.model_hyperparams import *


TASKS = {Task.MATH_STORY: 0,
                   Task.TOM: 1,
                   Task.MATCH_REL: 2,
                   Task.TWO_BK: 3,
                   Task.REWARD: 4,
                   Task.FACES_SHAPES: 5,
                   Task.T: 6
                   }





def test_time_to_validate(dir):
    for _ in range(1):
        dtfiles = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("dtseries.nii")]

        for f in dtfiles[:min(10, len(dtfiles))]:
            t, shape = time_ciftiopen(f)
            print ("{0}  time: {1:.6f}, shape: {2}".format(os.path.basename(f), t, shape))
    return

def time_ciftiopen(cifti_filepath):
    start = time.time()
    arr, (axis, bm) = open_cifti(cifti_filepath)
    stop = time.time()
    return (stop-start, arr.shape)

def convert_to_float32(file):
    arr = np.load(file)
    arr = arr.astype(np.float32)
    np.save(file, arr)

def stam_fsl():
    x = np.random.rand(5,3)
    y = np.dot(x, np.random.rand(3,1))
    y = np.random.rand(5,20) + 0.2 * y
    fsl_glm(x,y)

def compare(cifti, mat):
    arr1, (series, bm) = open_cifti(cifti)
    mat_arr = load_ndarray_from_mat(mat)
    if np.shape(arr1) != np.shape(mat_arr):
        arr1 = arr1.transpose()
    if np.shape(arr1) != np.shape(mat_arr):
        raise ValueError("arrays are not the same shape: {0} and {1}".format(arr1.shape, mat_arr.shape))
    if np.allclose(arr1, mat_arr):
        print("all equal")
        return True
    diff = abs(arr1 - mat_arr)
    print(np.count_nonzero(diff > 0.00001))
    return

def compare_ciftis(cifti1, cifti2, filters=None):
    arr1, (series, bm) = open_cifti(cifti1)
    arr2, (series2, bm2) = open_cifti(cifti2)
    if np.shape(arr1) != np.shape(arr2):
        n_vertices = min(np.size(arr1, axis= 1), np.size(arr2, axis= 1))
        arr1 = arr1[:, :n_vertices]
        arr2 = arr2[:, :n_vertices]
    if np.allclose(arr1, arr2):
        print("all equal")
        return True
    diff = abs(arr1 - arr2)
    n_diff = (np.count_nonzero(diff > 0.001))
    print('{} vertices are different'.format(n_diff))
    if n_diff<1000:
        for idx in (np.arange(59412)[np.squeeze(diff) > 0.001]):
            print(idx)
    corr = np.corrcoef(arr1, arr2)
    filters_in_p_jgfd = filters[np.squeeze(diff)>0,:]
    return
   # 'features_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
#r'C:\Users\ASUS\Dropbox\PITECA\Code\MATLAB\Features_firstSubj_firstSession.mat'

def check():
    ica_both_lowdim, (series, bm) = cifti.read(definitions.ICA_LOW_DIM_PATH)
    spatial_filters_raw, (series, bm) = cifti.read(definitions.ICA_LOW_DIM_PATH)
    spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDART_BM.CORTEX])
    soft_filters = softmax((spatial_filters_raw).astype(float) * TEMPERATURE)
    soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
    soft_filters[:, 2] = 0
    soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDART_BM.N_CORTEX, 1])
    f1 = r'D:\Projects\PITECA\Data_for_testing\predictions_nn\LANGUAGE\MATH_STORY\000071_LANGUAGE_MATH_STORY_predicted.dtseries.nii'
    f2 = r'D:\Projects\PITECA\Data_for_testing\predictions_nn\temp_071_LANGUAGE_MATH_STORY(2).dtseries.nii'
    compare_ciftis(f1,f2, soft_filters)

if __name__ == "__main__":
    check()
