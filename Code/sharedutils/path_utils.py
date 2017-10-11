import os

from sharedutils.constants import FEATS_EXT, PREDICT_OUTPUT_EXT, DTSERIES_EXT
from GUI.settings_controller import get_features_folder
from definitions import CANONICAL_CIFTI_DIR

"""
This utility gathers functions to specific PITECA string-path manipulation needs.
"""


def get_id(absolute_path):
    id = os.path.basename(absolute_path).split('_')[0]
    try:
        int(id) # validates that id is an integer
        return id
    except ValueError:
        return None


def get_features_path(id):
    filename = id + FEATS_EXT + DTSERIES_EXT
    return os.path.join(get_features_folder(), filename)


def extract_filenames(input_files_str):
    pathes = []
    filenames = input_files_str[1:-1].split(', ')
    for filename in filenames:
        start = filename.index("'")
        end = filename.rindex("'")
        pathes.append(filename[start+1:end])
    return pathes


def generate_file_name(outputpath, task, file_prefix):
    domain_outputpath = os.path.join(outputpath, task.domain().name)
    if not os.path.isdir(domain_outputpath):
        os.mkdir(domain_outputpath)
    task_outputpath = os.path.join(domain_outputpath, task.name)
    if not os.path.isdir(task_outputpath):
        os.mkdir(task_outputpath)
    return os.path.join(task_outputpath, file_prefix)


def generate_final_filename(filename):
    """
    Handles existing files with the same name.
    :param filename: the original file name (maybe exists already)
    :return: the final filename to be used to save the file (does not exits)
    """
    if not os.path.isfile(filename + DTSERIES_EXT):
        return filename
    i = 1
    while os.path.isfile(filename + "({})".format(i) + DTSERIES_EXT):
        i += 1
    return filename + "({})".format(i)

def get_canonical_path(task):
    return os.path.join(CANONICAL_CIFTI_DIR, 'canonical_{}.dtseries.nii'.format(task.full_name))
