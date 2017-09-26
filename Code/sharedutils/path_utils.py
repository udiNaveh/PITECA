from sharedutils.constants import FEATS_EXT, PREDICT_OUTPUT_EXT, DTSERIES_EXT
from GUI.settings_controller import get_features_folder
import os


def get_id(absolute_path):
    id = os.path.basename(absolute_path).split('_')[0]
    try:
        int(id) # validates that id is an integer
        return id
    except ValueError:
        # TODO: handle error @error_handling
        return None


def get_output_path(output_dir, id):
    filename = id + PREDICT_OUTPUT_EXT + DTSERIES_EXT
    return os.path.join(output_dir, filename)


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