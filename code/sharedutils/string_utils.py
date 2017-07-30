from sharedutils.constants import CONFIG_PATH, FEATS_EXT

def get_id(absolute_path):
    id = absolute_path.split('/')[-1].split('_')[0]
    try:
        int_id = int(id)
        return int_id
    except ValueError:
        ### TODO: handle error
        return None


def get_fetures_path(id):
    return CONFIG_PATH + "/" + id + FEATS_EXT
