import os
import errno


def all_same(iter, f):
    first = iter[0]
    return not any(f(x) != f(first) for x in iter)

def zeropad(i, length):
    assert type(i)==int
    i = str(i)
    n_zeros = length - len(str(i))
    if n_zeros>0:
        i = '0'*n_zeros + i
    return i


def union_dicts(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))

def safe_open(path, *args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, *args)
#
#
# def get_subject_to_feature_index_mapping(path):
#     mapping = {}
#     with open(path, 'r') as f:
#         for i in range(100):
#             subj_number = int(f.readline()) + 1
#             assert subj_number not in mapping
#             mapping[subj_number] = i
#     return mapping