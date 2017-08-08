import os


import unittest
import analysis.analyzer as analyzer
from sharedutils.subject import *


predictions_dir = r"D:\Projects\PITECA\Data\predictions"
actual_dir = r"D:\Projects\PITECA\Data\actual"
prediction_files = os.listdir(predictions_dir)
subjects = []
for filename in prediction_files:
    subject_id = filename[:6]
    if 'MATH_STORY' in filename and subject_id.isdigit():
        subjects.append(Subject(subject_id=subject_id,
                                predicted= {Task.MATH_STORY : os.path.join(predictions_dir,filename)},
                                actual= {Task.MATH_STORY : os.path.join(actual_dir,filename.replace("_predicted", ""))}))






class TestAnalyzer(unittest.TestCase):


    @unittest.skip("test_get_correlations skipped")
    def test_get_correlations(self):
        corrs, withmean, withcanonical = analyzer.get_predictions_correlations(subjects, Task.MATH_STORY, subjects[0].predicted[Task.MATH_STORY])
        self.assertTrue(np.allclose(np.diag(corrs), np.ones(len(subjects))))
        self.assertTrue(np.allclose(corrs[0,:], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withmean))
        return

    def test_get_predicted_actual_correlations(self):
        predicted_actual_correlations = analyzer.get_predicted_actual_correlations(subjects, Task.MATH_STORY)
        np.save('MATH_STORY_pred_actual_corrs_10subjects.npy', predicted_actual_correlations)
        for row in predicted_actual_correlations.tolist():
            print(row)



if __name__ == '__main__':
    unittest.main()
