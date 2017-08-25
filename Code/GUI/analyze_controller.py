from sharedutils import constants, path_utils, subject, dialog_utils
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtCore
from GUI.popups import analysis_working_dlg_controller
from GUI.analyze_working_thread import AnalysisWorkingThread, AnalysisTask
from GUI.graphics import graphics
import definitions


class AnalyzeController:

    def __init__(self, ui):
        self.ui = ui

    def __create_subjects(self, task, predicted_files_str, actual_files_str=None):
        '''
        Creates subjects with minimal info required for analysis.
        :param task: The task chosen for the analysis
        :param predicted_files_str: The input string from the user
        :param actual_files_str: The input string from the user (oprional)
        :return: a list of Subjects
        '''
        subjects = []

        predicted_files = path_utils.extract_filenames(predicted_files_str)
        for file in predicted_files:
            curr_subject = subject.Subject()
            curr_subject.subject_id = path_utils.get_id(file)
            curr_subject.predicted = {task: file}
            subjects.append(curr_subject)

        if actual_files_str:
            actual_files = path_utils.extract_filenames(actual_files_str)

            # Check predicted and actual match
            predicted_ids = [subject.subject_id for subject in subjects]
            actual_ids = [path_utils.get_id(file) for file in actual_files]
            if not set(predicted_ids) == set(actual_ids):
                dialog_utils.print_error("Predicted and actual don't match.")
                return
            else:
                # Add subjects the "actual" property
                for file in actual_files:
                    match_subject = next(
                        subject for subject in subjects if subject.subject_id == path_utils.get_id(file))
                    match_subject.actual = file

        return subjects

    def __handle_results(self, analysis_task, dlg, data):
        dlg.close()
        graphic_dlg = graphics.GraphicDlg(analysis_task, data)
        graphic_dlg.show()

    def update_tasks(self):
        self.ui.taskComboBox.clear()
        domain = constants.Domain[self.ui.domainComboBox.currentText()]
        self.ui.taskComboBox.addItems([task.name for task in domain.value])

    def onPredictedInputBrowseButtonClicked(self):
        dialog_utils.browse_files(self.ui.selectPredictedLineEdit)

    def onActualInputBrowseButtonClicked(self):
        dialog_utils.browse_files(self.ui.addActualLineEdit)

    def onRunAnalysisButtonClicked(self):
        predicted_files_str = self.ui.selectPredictedLineEdit.text()
        task = constants.Task[self.ui.taskComboBox.currentText()]
        subjects = self.__create_subjects(task, predicted_files_str)

        # Check all input provided
        if not predicted_files_str:
            dialog_utils.print_error("Please provide input.")
            return

        # Prepare additional analysis parameters
        analysis_task = None
        outputdir = definitions.ANALYSIS_DIR
        other_path = None

        if self.ui.analysisMeanRadioButton.isChecked():
            analysis_task = AnalysisTask.Analysis_Mean

        elif self.ui.analysisCorrelationsRadioButton.isChecked():
            analysis_task = AnalysisTask.Analysis_Correlations
            # TODO: add other_path = ...

        elif self.ui.analysisSignificantRadioButton.isChecked():
            # TODO: as I understand, there is not function for this in analyzer.
            # Remove this option from GUI or add a function to analyzer
            analysis_task = AnalysisTask.Analysis_Significance

        else:
            dialog_utils.print_error("Please choose analysis")
            return

        thread = AnalysisWorkingThread(analysis_task, subjects, task, outputdir, other_path)
        dlg = analysis_working_dlg_controller.AnalysisWorkingDlg()
        dlg.show()
        thread.progress_finished_sig.connect(lambda: self.__handle_results(analysis_task, dlg, thread.results))
        thread.start()

    def onRunComparisonButtonClicked(self):
        predicted_files_str = self.ui.selectPredictedLineEdit.text()
        actual_files_str = self.ui.selectPredictedLineEdit.text()
        subjects = self.__create_subjects(predicted_files_str, actual_files_str)
        task = constants.Task[self.ui.taskComboBox.currentText()]

        # Prepare additional analysis parameters
        analysis_task = None
        outputdir = constants.TMP_ANALYSIS_PATH
        other_path = None

        if self.ui.comparisonCorrelationsRadioButton.isChecked():
            analysis_task = AnalysisTask.Compare_Correlations

        elif self.ui.comparisonSignificantRadioButton.isChecked():
            analysis_task = AnalysisTask.Compare_Significance

        else:
            dialog_utils.print_error("Please choose a comparison functionality.")
            return

        thread = AnalysisWorkingThread(analysis_task, subjects, task, outputdir, other_path)
        dlg = analysis_working_dlg_controller.AnalysisWorkingDlg()
        dlg.show()
        thread.progress_finished_sig.connect(lambda: dlg.close())
        thread.start()
        # TODO: duplicated code with onAnalysisButtonClicked

