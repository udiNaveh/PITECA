import ast
import importlib.util
import os
import pickle
import time

import matplotlib.pyplot as plt

import definitions
from analysis.analyzer import *
from sharedutils.cmd_utils import show_maps_in_wb_view
from sharedutils.general_utils import inverse_dicts
from sharedutils.linalg_utils import *

tasks_file = os.path.join(definitions.LOCAL_DATA_DIR, 'HCP_200', "moreTasks.npy")
all_tasks_200s = r'D:\Projects\PITECA\Data_for_testing\time_series\AllTasks.npy'
all_tasks_200s_new_path = r'D:\Projects\PITECA\Data_for_testing\time_series\allTasksReordered.npy'

mapping_path100_to_200 = r'D:\Projects\PITECA\Data\docs\100_to_200'
all_features_path = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order.npy")

all_features_path_200 = os.path.join(r'D:\Projects\PITECA\Data', "all_features_only_cortex_in_order_200.npy")
all_features_path_mat = r'C:\Users\ASUS\Downloads\AllFeatures.mat'



#all_features_raw = np.load(all_features_path)



TASKS = [Task.MATH_STORY,
                   Task.TOM,
                   Task.MATCH_REL,
                   Task.TWO_BK,
                   Task.REWARD,
                   Task.FACES_SHAPES,
                   Task.T,
                   ]

def linear_weights_split():
    LINEAR_WEIGHTS_DIR = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'new')
    for task in TASKS:
        weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'learned_by_roi_70s',
                                    'linear_weights_{}.pkl'.format(
                                        task.full_name))
        weights = pickle.load(open(weights_path, 'rb'))


        weights_path = os.path.join(LINEAR_WEIGHTS_DIR, 'learned_by_roi_70s',
                                    'linear_weights_{}.pkl'.format(
                                        task.full_name))
        weights_by_task = {i: [weights[1:, i:i + 1], weights[0, i:i + 1]] for i in range(np.size(weights, axis=1)) if i != 2}
        save_pickle(weights_by_task, weights_path )
        print()


def explore_tasks():
    subjects = np.arange(0,200)
    tasks = np.load(all_tasks_200s_new_path)

    task_language = tasks[0, :, :STANDARD_BM.N_CORTEX]
    task_mean = np.mean(task_language, axis=1)
    task_std = np.std(task_language, axis=1)
    task_max  = np.percentile(task_language, 95, axis=1)
    task_min = np.percentile(task_language, 5, axis=1)
    mean_all = np.mean(task_language)
    std_all = np.std(task_language)

    plt.scatter(subjects, task_min)
    plt.scatter(subjects, task_max)
    plt.scatter(subjects, task_mean)
    plt.show()
    print("")

def open_canonical_files():
    tasks_ciftis_path = r'D:\Projects\PITECA\Data_for_testing\actual_task_activation'
    file_names = [os.path.join(tasks_ciftis_path, 'canonical_{}.dtseries.nii'.format(task.full_name)) for task in TASKS]
    show_maps_in_wb_view(file_names)

def create_tasks_files():
    tasks_ciftis_path = r'D:\Projects\PITECA\Data_for_testing\actual_task_activation'
    tasks = np.load(all_tasks_200s)
    full_bm = pickle.load(open(definitions.BM_FULL_PATH, 'rb'))
    for i, task in enumerate(TASKS):
        mean_data = np.reshape(np.mean(tasks[i, :,:], axis=0), [1, STANDARD_BM.N_TOTAL_VERTICES])
        f_name = 'canonical_{}'.format(task.full_name)
        save_to_dtseries(os.path.join(tasks_ciftis_path, f_name), full_bm, mean_data)
        print('saved canonical for {}'.format(task.full_name))
    return

def create_features_files():
    path2dir = r'D:\Projects\PITECA\Data\extracted features\features from mat'
    subjects = ['100307', '101107', '102816', '105014', '105216']
    cortex_bm = pickle.load(open(definitions.BM_CORTEX_PATH, 'rb'))
    features_200_first_5 = np.load(all_features_path_200)[:,[100,0,101,102,103],:]
    for j in range(len(subjects)):
        data = features_200_first_5[:, j, :]
        data = data.transpose()
        filename = os.path.join(path2dir, '{}_features'.format(subjects[j]))
        save_to_dtseries(filename, cortex_bm, data)
        print('saved {}'.format(os.path.basename(filename)))


def load_tasks():
    tasks200 = np.load(all_tasks_200s)
    tasks100 = np.load(tasks_file)
    map_100_to_200 = {}
    with open(mapping_path100_to_200) as f:
        for line in f:
            a, b = line.split(':')
            a = int(a);
            b = int(b)
            map_100_to_200[a] = b

    map_200_to_100 = {b: a for a, b in map_100_to_200.items()}
    tasks_200_new = np.zeros_like(tasks200)
    tasks_200_new[:,:100,:] = tasks100
    tasks_200_new[:, 100:, :] = tasks200[:,[i for i in range(200) if i not in map_200_to_100],:]

def save_new_tasks():
    map_100_to_200 = {}
    with open(mapping_path100_to_200) as f:
        for line in f:
            a, b = line.split(':')
            a = int(a);
            b = int(b)
            map_100_to_200[a] = b
    map_200_to_100 = {b: a for a, b in map_100_to_200.items()}
    only_in_200_idxs = [i for i in range(200) if i not in map_200_to_100]

    tasks200 = np.load(all_tasks_200s)
    tasks_200_new = np.zeros_like(tasks200)
    tasks_200_new[:,:100,:] = tasks200[:,[map_100_to_200[i] for i in range(100)],:]
    tasks_200_new[:, 100:, :] = tasks200[:, only_in_200_idxs, :]
    orig = np.load(tasks_file)
    for i in range(100):
        print(np.max(np.abs(tasks_200_new[:,i,:] - orig[:,i,:])))
    np.save(all_tasks_200s_new_path, tasks_200_new)


def test_ratio():
    features_200_first100 = np.load(all_features_path_200)[:,:100,:]
    features_100 = np.load(all_features_path)
    for i in range(100):
        a = demean_and_normalize(features_100[:,i,:])
        b = demean_and_normalize(features_200_first100[:,i,:])
        print (np.max(np.abs(a-b)), np.mean(np.abs(a-b)))

def test_correlations(path_to_tasks_dir, path_to_actual_dir):

    subjects_ids = ['100307', '101107', '102816', '105014', '105216']
    subjects = []
    for sid in subjects_ids:
        s = Subject()
        s.subject_id = sid
        s.output_dir = path_to_tasks_dir
        s.predicted = {t : s.get_predicted_task_filepath(t).replace('(1)','')+'.dtseries.nii' for t in AVAILABLE_TASKS}
        s.actual = {t: s.get_actual_task_filepath(t, path_to_actual_dir).replace('(1)','')+'.dtseries.nii' for t in AVAILABLE_TASKS}
        subjects.append(s)
    for task in AVAILABLE_TASKS:
        corr = np.diag(get_predicted_actual_correlations(subjects, task))
        print(task.full_name, corr)

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

def get_classes_from_file(file_path):
    if not str.endswith(file_path, '.py'):
        return []




def show_info(functionNode):
    print("Function name:", functionNode.name)
    print("Args:")
    for arg in functionNode.args.args:
        #import pdb; pdb.set_trace()
        print("\tParameter name:", arg.arg)

def do_show_info():
    filename = os.path.join(definitions.CODE_DIR, 'model', 'models.py')
    with open(filename) as file:
        node = ast.parse(file.read())

    functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

    for function in functions:
        show_info(function)

    for class_ in classes:
        print("Class name:", class_.name)
        print("Class bases:", class_.bases)
        methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
        for method in methods:
            show_info(method)

def xxx():
    location_path = os.path.join(definitions.CODE_DIR, 'model')

    spec = importlib.util.spec_from_file_location('models.py', location=location_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    foo.MyClass()

def read_results_text_file_and_plot(path):
    results = {}
    current_model_name = ''
    current_task_name = ''
    with open(path) as f:
        for line in f:
            if line.isspace():
                continue
            elif '##' in line:
                break
            else:
                line = line.strip()
                if '=' not in line:
                    current_model_name = line[:3] +'_' + line[-5:]
                    results[current_model_name] = {}
                else:
                    rh_lh = line.split('=')
                    rh = rh_lh[0].strip(); lh = rh_lh[1].strip()
                    if rh == 'task':
                        current_task_name = lh.split('.')[-1]
                        results[current_model_name][current_task_name] = {}
                    else:

                        results[current_model_name][current_task_name][rh] = float(lh)




    results = inverse_dicts(results)

    for task in results:
        models_names  = [k for k in results[task].keys()]
        models_names = sorted(models_names, key = lambda  name : results[task][name]['mean corr with self'])
        corrs = [results[task][k]['mean corr with self'] for k in models_names]
        corrs_with_mean = [results[task][k]['mean corr with other'] for k in models_names]

        plotchartsoverlap(corrs, corrs_with_mean, [name for name in models_names], title=task)




    print("")



def plotchartsoverlap(correlation, avg_corr_with_mean, model_names, title):

    n_groups = 5
    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, correlation, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='corr-self')

    rects2 = plt.bar(index + bar_width, avg_corr_with_mean, bar_width,
                     alpha=opacity,
                     color='c',
                     error_kw=error_config,
                     label='corr-other')

    plt.xlabel('model name')
    plt.ylabel('Pearson correlation')

    plt.ylim([0.2, 0.8])
    plt.yticks(np.linspace(0.2,0.8,13))
    plt.xticks(index + bar_width / 2, model_names, rotation='horizontal')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
    #plt.savefig("{}.png".format(title))



    return


def check_preds_are_different():
    pathdir = r'C:\Users\ASUS\PycharmProjects\PITECA\Data\Predictions\WM\TWO_BK\temp'
    filename_format = '000075_WM_TWO_BK_predicted{}.dtseries.nii'
    other_filename_format = '000074_WM_TWO_BK_predicted{}.dtseries.nii'
    arrs = []
    for i in range(1):
        arr, (ax, bm) = open_1d_file(os.path.join(pathdir, filename_format.format(i)))
        arrs.append(arr)
    arr, (ax, bm) = open_1d_file(os.path.join(pathdir, other_filename_format.format(0)))
    arrs.append(arr)

    d = np.abs(arrs[0] - arrs[1])
    print()



if __name__ == "__main__":
    linear_weights_split()
    #results_path = r'D:\Projects\PITECA\Data\docs\7030_models_results_corrected_test'
    #read_results_text_file_and_plot(results_path)