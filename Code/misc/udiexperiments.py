from sharedutils.io_utils import *
from sharedutils.subject import *
import time
import os

BASE = r"D:\Projects\PITECA\Data"
os.path.join(r"D:\Projects\PITECA\Data")
ACTUAL = os.path.join(r"D:\Projects\PITECA\Data",'actual')
FEATURES = os.path.join(r"D:\Projects\PITECA\Data",'extracted features')
PREDICTIONS = os.path.join(r"D:\Projects\PITECA\Data",'predictions')


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
    all_correlations = np.load(r"C:\Users\ASUS\Dropbox\PITECA\Data\regression_model\loo_betas_7_tasks\all_correlations.npy")
    print(all_correlations[:10,:10])