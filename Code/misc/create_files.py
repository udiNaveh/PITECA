import numpy as np
import os

import definitions
from sharedutils.io_utils import open_cifti, open_mat_file, load_ndarray_from_mat
from sharedutils.constants import STANDART_BM



def create_pinvG():
    arr, (series, bm) = open_cifti(definitions.ICA_LR_MATCHED_PATH)
    arr = np.transpose(arr)
    G = np.zeros([STANDART_BM.N_TOTAL_VERTICES, 76])
    G[:STANDART_BM.N_LH, :38] = arr[:STANDART_BM.N_LH, :]
    G[STANDART_BM.N_LH : STANDART_BM.N_CORTEX, 38:76] = arr[STANDART_BM.N_LH : STANDART_BM.N_CORTEX, :]
    pinvG = np.linalg.pinv(G)
    pinvGT = pinvG.transpose()
    np.save(os.path.join(definitions.DATA_DIR, "pinvg"), pinvGT)

def convert_to_float32(file):
    arr = np.load(file)
    arr = arr.astype(np.float32)
    np.save(file, arr)

def cifti2npy(path,outpath ,transpose=False):
    arr, (ser, bm)= open_cifti(path)
    if transpose:
        arr = arr.transpose().astype(np.float32)
    np.save(outpath, arr)

if __name__ == '__main__':
    cifti2npy(definitions.SC_CLUSTERS_PATH, definitions.SC_CLUSTERS_PATH+".npy", transpose=True)
    print("")