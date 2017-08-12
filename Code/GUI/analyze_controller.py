from sharedutils import gui_utils, subject, constants
from analysis import analyzer
from sharedutils import constants, path_utils, subject


class AnalyzeController:

    def __init__(self, ui):
        self.ui = ui
        self.outputdir = constants.TMP_ANALYSIS_PATH

    def __create_subjects(self, predicted_files_str, actual_files_str=None):
        subjects = []
        predicted_files = path_utils.extract_filenames(predicted_files_str)
        # TODO: complete after we change predicted and actual type
        return subjects

    def update_tasks(self):
        self.ui.taskComboBox.clear()
        domain = constants.Domain[self.ui.domainComboBox.currentText()]
        self.ui.taskComboBox.addItems([task.name for task in domain.value])

    def onPredictedInputBrowseButtonClicked(self):
        gui_utils.browse_files(self.ui.selectPredictedLineEdit)

    def onActualInputBrowseButtonClicked(self):
        gui_utils.browse_files(self.ui.addActualLineEdit)

    def onRunAnalysisButtonClicked(self):
        predicted_files_str = self.ui.selectPredictedLineEdit.text()
        subjects = self.__create_subjects(predicted_files_str)
        task = constants.Task[self.ui.taskComboBox.currentText()]

        # Check all input provided
        if not predicted_files_str or not task:
            # TODO: @error_handling
            pass

        if self.ui.analysisMeanRadioButton.isChecked():
            analyzer.get_prediction_mean(subjects, task, self.outputdir)
        elif self.ui.analysisCorrelationsRadioButton.isChecked():
            analyzer.get_predictions_correlations(subjects, task, other_path=None) # TODO: get other path from somewhere...
        elif self.ui.analysisSignificantRadioButton.isChecked():
            pass # TODO: as I understand, there is not function for this in analyzer.
                 # Remove this option from GUI or add a function to analyzer
        else:
            pass
            # TODO: @error_handling

    def onRunComparisonButtonClicked(self):
        predicted_files_str = self.ui.selectPredictedLineEdit.text()
        actual_files_str = self.ui.selectPredictedLineEdit.text()
        subjects = self.__create_subjects(predicted_files_str, actual_files_str)
        task = constants.Task[self.ui.taskComboBox.currentText()]

        if self.ui.comparisonCorrelationsRadioButton.isChecked():
            analyzer.get_predicted_actual_correlations(subjects, task)
        elif self.ui.comparisonSignificantRadioButton.isChecked():
            analyzer.get_significance_overlap_maps_for_subjects(subjects, task, self.outputdir)
        else:
            pass
        # TODO: @error_handling
