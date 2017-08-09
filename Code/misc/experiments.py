import os.path
from py_matlab_utils import *
import matlab.engine
from scipy.signal import detrend
import numpy as np
from constants import *
import cifti
import io_utils
import linalg_utils
from py_matlab_utils import params_to_json
import Package1.methods

'''
this module is to be used to run small tests and check ups when we write code, or just want to run a one-time 
piece of code.
It is better to have all the 'garbage' in one file, instead of spreading it across
different files. So the best practice will be that other files in the directory are not runnable 
(no main thread, all code is either constants or function definitions). That way we can keep the real code
relatively clean.
Note though that this is not instead of real, robust unit tests that we will have to write and use.



'''




outfilename = 'params.json'
rfmri_example_filename = 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'


def create_param_file2():
    params = {}
    params['DATA_DIR'] = DATA_DIR
    params['ica_path'] = path.join(DATA_DIR, 'HCP_200', 'ica_LR_MATCHED.dtseries.nii')
    params['subcortex_path'] = path.join(DATA_DIR, 'HCP_200', 'SC_clusters.dtseries.nii')
    params['subjects'] = [{"ID" : '123', "input_file" : "", "features_file" : ""}]
    params['extras'] = path.join(MATLAB_CODE_DIR, 'extras')

    params_to_json('params.json', params)


def create_params_file():
    params = {}
    params['inputdir'] = r'C:\Users\ASUS\Dropbox\PITECA\Data'
    params['outputdir'] = r'C:\Users\ASUS\Dropbox\PITECA\Data\output'
    keys, vals = zip(*{'123456_fmri.dtseries.nii' : '123456_fmri_features.dtseries.nii',
                                    '123457_fmri.dtseries.nii' : '123457_fmri_features.dtseries.nii',
                                    '123458_fmri.dtseries.nii': '123458_fmri_features.dtseries.nii'}.items())
    params['subjects'] = []
    nums = [123,145,1764,197, 3542]
    for num in nums:
        subf = {}
        subf['id'] = str(num)
        subf['input'] = str(num)+ '_' + 'input'
        subf['output'] = str(num) + '_' + 'output'
        subf['features'] = str(num) + '_' + 'features'
        params['subjects'].append(subf)

    params['input_output_files'] = (keys,vals)
    params['threshold'] = 2.1
    params['nvertices'] = 92342
    params_to_json( r'C:\Users\ASUS\Dropbox\PITECA\Data\params3.json', params)


def test_linalgutils_on_real_data():
    print(1)
    input_file = os.path.join(PATHS.DATA_DIR, rfmri_example_filename)
    arr, axes = cifti.read(input_file)
    features_001 = np.load('features_001.npy')
    arr = np.transpose(features_001)

    cifti.write('features_001.dtseries.nii', arr, axes)
    return 0


def get_betas():
    betas_dir = os.path.join(PATHS.DATA_DIR, 'regression_model', 'loo_betas')


def make_smaller_file():
    dscalar = \
    [os.path.join(PATHS.WB_TUTORIAL_DATA, f) for f in os.listdir(PATHS.WB_TUTORIAL_DATA) if 'scalar' in f][0]
    mat, (axis, BM) = cifti.read(dscalar)
    #new_arr = [v for v in BM.arr if str.startswith(v[2], 'CIFTI_STRUCTURE_CORTEX')]
    BM.arr = BM.arr[STANDART_BM.CORTEX]

    axis.arr = axis.arr[:2]
    axis.arr[0][0] =  "task1"
    axis.arr[1][0] = "task2"
    mat = mat[:2, :len(BM.arr)]
    cifti.write(os.path.join(PATHS.GARBAGE,"smaller_BM_two_tasks.dscalar.nii"), mat, (axis, BM))


def convert_data_to_numpy(dir):
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isfile(full_path) and f.endswith(".mat"):

            full_path_npy = full_path.replace('.mat', '')
            arr = load_ndarray_from_mat(full_path)
            np.save(full_path_npy, arr)


def convert_to_float32(file):
    arr = np.load(file)
    arr = arr.astype(np.float32)
    np.save(file, arr)


def test1():
    spatial_filters_path =  os.path.join(PATHS.DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')
    arr, (series, bm) = io_utils.open_cifti(spatial_filters_path)
    print("h")



if __name__ == "__main__":
    #convert_data_to_numpy(PATHS.DATA_DIR)
    test1()
