import unittest
from os.path import join as join_path

from model.models import LinearModel as LinearModel
from sharedutils.subject import *
from sharedutils.general_utils import *
from sharedutils.constants import *


OUTPUTPATH = r"D:\Projects\PITECA\Data\predictions"





class TestModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_linear_model(self):
        subjects = create_subjects(range(1,10), PATHS.EXTRACTED_FEATURES_PATH, OUTPUTPATH)
        lmodel = LinearModel([Task.MATH_STORY, Task.MATCH_REL])
        for subj in subjects:
            lmodel.predict(subj)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
