from PyQt5.QtCore import QThread
from PyQt5 import QtCore
import time, math
from analysis import analyzer
from enum import Enum


class AnalysisTask(Enum):
    Analysis_Mean = 1
    Analysis_Correlations = 2
    Analysis_Significance = 3

    Compare_Correlations = 4
    Compare_Significance = 5


class AnalysisWorkingThread(QThread):

    progress_finished_sig = QtCore.pyqtSignal()

    def __init__(self,
                 analysis_task,  # type: AnalysisTask
                 subjects,  # type: list of Subject
                 task,  # type: Task
                 outputdir,  # type: string (This class assumes it is a valid path)
                 other_path=None,  # type: string (This class assumes it is a valid path)
                 parent=None):
        super(AnalysisWorkingThread, self).__init__(parent)

        # Set parameters needed for analysis methods
        self.analysis_task = analysis_task
        self.subjects = subjects
        self.task = task
        self.outputdir = outputdir
        self.other_path = other_path

    def __del__(self):
        self.wait()

    def run(self):

        if self.analysis_task == AnalysisTask.Analysis_Mean:
            analyzer.get_prediction_mean(self.subjects, self.task, self.outputdir)

        elif self.analysis_task == AnalysisTask.Analysis_Correlations:
            analyzer.get_predictions_correlations(self.subjects, self.task, self.other_path)

        elif self.analysis_task == AnalysisTask.Analysis_Significance:
            pass
            # TODO: what is the function in analyzer for this?

        elif self.analysis_task == AnalysisTask.Compare_Correlations:
            analyzer.get_predicted_actual_correlations(self.subjects, self.task)

        elif self.analysis_task == AnalysisTask.Compare_Significance:
            analyzer.get_significance_overlap_maps_for_subjects(self.subjects, self.task, self.outputdir)

        self.quit()


