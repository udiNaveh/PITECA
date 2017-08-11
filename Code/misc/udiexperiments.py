from sharedutils.io_utils import *
from sharedutils.subject import *
import time
import os



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


if __name__ == "__main__":
    pass