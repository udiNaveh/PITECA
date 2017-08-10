from sharedutils.constants import TMP_FEATURES_PATH, FEATS_EXT, PREDICT_OUTPUT_EXT, DTSERIES_EXT
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
    return os.path.join(TMP_FEATURES_PATH, filename)


