import os
import cifti
from sharedutils.subject import  *
from sharedutils.io_utils import  *
from sharedutils.constants import *
import sharedutils.general_utils as general_utils
from model.models import LinearModel


all_features_path = os.path.join(r'D:\Projects\PITECA\Data',"all_features.npy")
ica_both_lowdim_path = os.path.join(PATHS.DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')
tasks_path = os.path.join(PATHS.DATA_DIR, 'moreTasks.npy')

def create_features_for_learning():
    all_features = np.load(all_features_path)
    shape = np.shape(all_features)
    assert shape[0] == STANDART_BM.N_TOTAL_VERTICES
    n_features = shape[2]
    n_subjects = shape[1]
    bm = load_thin_bm()
    with open('subjects_features.txt', 'r') as f:
        for i in range(100):
            subj_number = int(f.readline()) + 1
            filename = general_utils.zeropad(subj_number, 6) + "_features.dtseries.nii"
            filename = os.path.join(r'D:\Projects\PITECA\Data', filename)
            subj_features = all_features[STANDART_BM.CORTEX, i, :]
            save_to_dtseries(filename, bm, np.transpose(subj_features))



    return


def load_thin_bm():
    bmpath = os.path.join(PATHS.DATA_DIR, 'garbage', 'smaller_BM_two_tasks.dscalar.nii')
    data, (s, bm) = cifti.read(bmpath)
    return bm


def load_full_bm():
    bmpath = os.path.join(PATHS.DATA_DIR, 'garbage', 'full_BM.dscalar.nii.dtseries.nii')
    data, (s, bm) = cifti.read(bmpath)
    return bm


def create_actual_task_files():
    actual_path = r"D:\Projects\PITECA\Data\actual"
    bm = load_full_bm()
    tasks = np.load(tasks_path)
    task_names = sorted(LinearModel.available_tasks.keys(), key= lambda key : LinearModel.available_tasks[key])

    for subj_index in range(1,30):
        subject = Subject(subject_id=general_utils.zeropad(subj_index + 1, 6),
                          output_path=actual_path)
        for task_index in range(len(task_names)):
            mat = tasks[task_index, subj_index,:]
            filename = (subject.predicted_task_filepath(task_names[task_index])).replace("_predicted","")
            save_to_dtseries(filename, bm, mat)





    return


if __name__ == "__main__":
    #create_actual_task_files()
    pass