import unittest
from os.path import join as join_path

from model.models import LinearModel as LinearModel
from model.models import FeatureExtractor as FeatureExtractor
from sharedutils.subject import *
from sharedutils.general_utils import *
from sharedutils.constants import *
import definitions
from sharedutils.io_utils import *

OUTPUTPATH = definitions.PREDICTIONS_DIR

LOCAL_DATA_DIR = r'D:\Projects\PITECA\Data'
LOCAL_FEATURES_DIR = join_path(LOCAL_DATA_DIR, 'extracted features')
input_timeseries = join_path(LOCAL_DATA_DIR,'rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii')
features_path = join_path(definitions.EXTRACTED_FEATURES_DIR, 'extracted_features_'+'rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii')


AllFeatures_File = os.path.join(r'D:\Projects\PITECA\Data',"all_features.npy")
all_features = np.load(AllFeatures_File)


class TestModel(unittest.TestCase):


        



    @unittest.skip("skip")
    def test_feature_extractor(self):
        subject = Subject(subject_id=123456, output_dir=OUTPUTPATH,
                          input_path=input_timeseries, features_path= features_path)
        fe = FeatureExtractor()
        features = fe.get_features(subject)



if __name__ == '__main__':
    unittest.main()
