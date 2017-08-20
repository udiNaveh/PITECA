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

input_timeseries = join_path(LOCAL_DATA_DIR,'rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii')
features_path = join_path(definitions.EXTRACTED_FEATURES_DIR, 'extracted_features_'+'rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii')



class TestModel(unittest.TestCase):

    def setUp(self):
        pass
    @unittest.skip("skip")
    def test_linear_model(self):
        subjects = create_subjects(range(1, 10), definitions.EXTRACTED_FEATURES_DIR, OUTPUTPATH)
        lmodel = LinearModel([Task.MATH_STORY, Task.MATCH_REL])
        for subj in subjects:
            lmodel.predict(subj)
        self.assertEqual(True, True)

    def test_feature_extractor(self):
        subject = Subject(subject_id=123456, output_dir=OUTPUTPATH,
                          input_path=input_timeseries, features_path= features_path)
        fe = FeatureExtractor()
        features = fe.get_features(subject)



if __name__ == '__main__':
    unittest.main()
