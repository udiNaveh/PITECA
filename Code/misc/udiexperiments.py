from sharedutils.io_utils import *
from sharedutils.subject import *
from sharedutils.linalg_utils import *
import time
import os

import definitions



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

def compare_ciftis(cifti1, cifti2):
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
    print(np.count_nonzero(diff > 0.001))
    corr = np.corrcoef(arr1, arr2)
    return
   # 'features_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
#r'C:\Users\ASUS\Dropbox\PITECA\Code\MATLAB\Features_firstSubj_firstSession.mat'


if __name__ == "__main__":
    features1 = os.path.join(definitions.EXTRACTED_FEATURES_DIR, 'extracted_features_rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii')
    features2 = os.path.join(definitions.EXTRACTED_FEATURES_DIR, 'extracted_features_rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii')
    compare_ciftis(features1, features2)
