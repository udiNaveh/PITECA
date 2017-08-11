import unittest
from os.path import join as join_path

from model.models import LinearModel as LinearModel
from sharedutils.subject import *
from sharedutils.general_utils import *
from sharedutils.constants import *
import definitions

OUTPUTPATH = definitions.PREDICTIONS_DIR





class TestModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_linear_model(self):
        subjects = create_subjects(range(1, 10), definitions.EXTRACTED_FEATURES_DIR, OUTPUTPATH)
        lmodel = LinearModel([Task.MATH_STORY, Task.MATCH_REL])
        for subj in subjects:
            lmodel.predict(subj)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
