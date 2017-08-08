from sharedutils.io_utils import *
import time
import os


os.path.join(r"D:\Projects\PITECA\Data")


def test_time_to_validate():
    dir = PATHS.HEAVY_DATA_DIR
    for _ in range(1):
        for f in os.listdir(dir):
            full_path = os.path.join(dir, f)
            if os.path.isfile(full_path) and f.endswith("dtseries.nii"):
                t, shape = time_ciftiopen(full_path)
                print ("{0}: {")
    return



def time_ciftiopen(cifti_filepath):
    start = time.time()
    arr, (axis, bm) = open_cifti(cifti_filepath)
    stop = time.time()
    return (stop-start, arr.shape)