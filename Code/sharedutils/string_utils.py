from sharedutils.constants import CONFIG_PATH, FEATS_EXT

def get_id(absolute_path):
    id = absolute_path.split("/")[-1].split('_')[0]
    try:
        int_id = int(id)
        return int_id
    except ValueError:
        ### TODO: handle error
        return None

def get_fetures_path(id):
    return CONFIG_PATH + "/" + id + FEATS_EXT

def get_full_path(dir, file_name):
    return dir + "/" + file_name


def zeropad(i, length):
    assert type(i)==int
    i = str(i)
    n_zeros = length - len(str(i))
    if n_zeros>0:
        i = '0'*n_zeros + i
    return i
