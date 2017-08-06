import os


import unittest
import analysis.analyzer as analyzer
from sharedutils.subject import *


predictions_dir = r"D:\Projects\PITECA\Data\predictions"
prediction_files = os.listdir(predictions_dir)
subjects = []
for filename in prediction_files:
    subject_id = filename[:6]
    if 'MATH_STORY' in filename:
        subjects.append(Subject(subject_id=subject_id,
                                predicted= {Task.MATH_STORY : os.path.join(predictions_dir,filename)}))






class TestAnalyzer(unittest.TestCase):



    def test_get_correlations(self):
        corrs, withmean, withcanonical = analyzer.get_predictions_correlations(subjects, Task.MATH_STORY, subjects[0].predicted[Task.MATH_STORY])
        self.assertTrue(np.allclose(np.diag(corrs), np.ones(len(subjects))))
        self.assertTrue(np.allclose(corrs[0,:], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withmean))


if __name__ == '__main__':
    unittest.main()
