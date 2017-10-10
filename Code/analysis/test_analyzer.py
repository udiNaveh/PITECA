import os


import unittest
import analysis.analyzer as analyzer
from sharedutils.subject import *


predictions_dir = os.path.join(definitions.PREDICTIONS_DIR, 'LANGUAGE', 'MATH_STORY')
actual_dir = os.path.join(definitions.DATA_DIR, 'Tasks')
overlap_dir = os.path.join(definitions.ANALYSIS_DIR, 'Overlap Maps')

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

    @unittest.skip("test_get_correlations skipped")
    def test_get_predicted_actual_correlations(self):
        predicted_actual_correlations = analyzer.get_predicted_actual_correlations(subjects, Task.MATH_STORY)
        #np.save('MATH_STORY_pred_actual_corrs_10subjects.npy', predicted_actual_correlations)
        for row in predicted_actual_correlations.tolist():
            print(row)

    @unittest.skip("checked")
    def test_correlations_both_functions(self):
        subs = subjects[:3]

        corrs, withmean, withcanonical = analyzer.get_predictions_correlations(subs, Task.MATH_STORY, subs[0].predicted[Task.MATH_STORY])
        self.assertTrue(np.allclose(np.diag(corrs), np.ones(len(subs))))
        self.assertTrue(np.allclose(corrs[0,:], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withmean))
        for s in subs:
            s.actual = s.predicted
        predicted_actual_correlations = analyzer.get_predicted_actual_correlations(subs, Task.MATH_STORY)
        self.assertTrue(np.allclose(corrs, predicted_actual_correlations))
        return

    @unittest.skip("done")
    def test_correlations_both_functions(self):
        subs = subjects

        corrs, withmean, withcanonical = analyzer.get_predictions_correlations(subs, Task.MATH_STORY, subs[0].predicted[Task.MATH_STORY])
        self.assertTrue(np.allclose(np.diag(corrs), np.ones(len(subs))))
        self.assertTrue(np.allclose(corrs[0,:], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withcanonical))
        self.assertFalse(np.allclose(corrs[1, :], withmean))
        predicted_actual_correlations = analyzer.get_predicted_actual_correlations(subs, Task.MATH_STORY)
        self.assertGreater(np.mean(np.diag(predicted_actual_correlations)), np.mean(predicted_actual_correlations)+0.05)


    @unittest.skip("not relevant")
    def test_values_of_prediction(self):
        for kvp in analyzer.get_predicted_task_maps_by_subject(subjects[:3], Task.MATH_STORY).items():
            print(kvp[0].subject_id)
            arr = kvp[1]
            print (np.mean(arr), np.max(arr), np.min(arr), np.median(arr), np.std(arr))
        for kvp in analyzer.get_actual_task_maps_by_subject(subjects[:3], Task.MATH_STORY).items():
            print(kvp[0].subject_id)
            arr = kvp[1]
            print (np.mean(arr), np.max(arr), np.min(arr), np.median(arr), np.std(arr))
            arr = arr[0, :STANDARD_BM.N_CORTEX]
            print(np.mean(arr), np.max(arr), np.min(arr), np.median(arr), np.std(arr))

    def test_get_significance_overlap_basic(self):
        arr1 = np.arange(9,0,-1)
        arr2 = np.array([1,0,-1]*3)
        overlap_map, iou_pos, iou_neg, iou_both = analyzer.get_significance_overlap_map(arr1, arr2, lambda a : analyzer.get_significance_thresholds(a, 35))
        self.assertAlmostEqual(iou_both, 0.25)
        self.assertAlmostEqual(iou_pos, 0.2)
        self.assertAlmostEqual(iou_neg, 0.2)
        self.assertTrue(all([ 4  ,1 ,-2 , 3,  0, -3,  2 ,-1, -4]==overlap_map))


    def test_get_significance_overlap_maps_for_subjects(self):
        analyzer.get_significance_overlap_maps_for_subjects(subjects[2:5], Task.MATH_STORY, overlap_dir)







if __name__ == '__main__':
    unittest.main()
