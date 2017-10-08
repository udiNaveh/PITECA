import os

from PyQt5.QtCore import Qt

from sharedutils import constants, path_utils, subject, dialog_utils
import GUI.globals as gb
from GUI.popups import analysis_working_dlg_controller
from GUI.analyze_working_thread import AnalysisWorkingThread, AnalysisTask
from GUI.graphics import graphics
import definitions
from GUI.settings_controller import get_analysis_results_folder


class AnalyzeController:
    """
    A class to control the GUI logic of the Analysis tab
    """

    global should_exit_on_error

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
            if curr_subject.subject_id == None:
                dialog_utils.inform_user(constants.NAMING_CONVENTION_ERROR)
                return []
            curr_subject.predicted = {task: file}
            subjects.append(curr_subject)

        if actual_files_str:
            actual_files = path_utils.extract_filenames(actual_files_str)

            # Check predicted and actual match
            predicted_ids = [subject.subject_id for subject in subjects]
            actual_ids = [path_utils.get_id(file) for file in actual_files]
            if not set(predicted_ids) == set(actual_ids):
                dialog_utils.print_error("The files provided as the actual activation do not match the predicted files.")
                return None
            else:
                # Add subjects the "actual" property
                for file in actual_files:
                    match_subject = next(
                        subject for subject in subjects if subject.subject_id == path_utils.get_id(file))
                    match_subject.actual = {task: file}

        # assert only unique subjects
        ids = [subject.subject_id for subject in subjects]
        if len(ids) != len(set(ids)):
            dialog_utils.print_error("Duplicate subjects. Please make sure to have only unique subjects in Analysis")
            return None
        return subjects

    def __handle_results(self, analysis_task, dlg, data, subjects):
        """
        The function to be called after the analysis progress is done.
        For each analysis task, handle the results in a different way.
        :param analysis_task: The task chosen by the user (mean/correlations/significance).
        :param dlg: The progress dialog to be closed.
        :param data: The results returned from the analyzer.
        :param subjects: A list of Subject(s), their data was analyzed.
        """
        dlg.close()
        gb.should_exit_on_error = False

        if analysis_task == AnalysisTask.Analysis_Mean:

            analysis_folder = get_analysis_results_folder()
            folder_path = os.path.join(analysis_folder, self.task.domain().name, self.task.name)

            dialog_utils.report_results("Done! Analysis result is saved in {}".format(folder_path),
                                        folder_path, gb.curr_cifti_filename)

        elif analysis_task in [AnalysisTask.Analysis_Correlations, AnalysisTask.Compare_Correlations, AnalysisTask.Compare_Significance]:

            if analysis_task == AnalysisTask.Compare_Significance:

                if len(subjects) == 1:
                    filepath = gb.curr_cifti_filename
                else:
                    filepath = None

                analysis_folder = get_analysis_results_folder()
                folder_path = os.path.join(analysis_folder, self.task.domain().name, self.task.name)

                dialog_utils.report_results("Done! Comparison result is saved in {}".format(folder_path),
                                            folder_path, filepath, False)
                title_base = "Intersection over Union of subjects overlap maps"

            if analysis_task == AnalysisTask.Analysis_Correlations:
                title_base = "Correlations between subjects' predictions"

            if analysis_task == AnalysisTask.Compare_Correlations:
                title_base = "Correlations between actual and predicted activation"

            title = "{}\n Domain: {} Task: {}".format(
                title_base, self.task.domain().name, self.task.name
            )
            graphic_dlg = graphics.GraphicDlg(analysis_task, data, subjects, title)
            graphic_dlg.setWindowModality(Qt.ApplicationModal)
            graphic_dlg.show()

        else:
            raise Exception('Unsupported action.')

            gb.should_exit_on_error = True


    def __handle_unexpected_exception(self, dlg, thread):
        """
        The function to be called when unhandled exception occurs in param thread.
        Terminates the thread, closing the progress dialog and pops up an error message.
        :param dlg: Progress dialog, to be closed.
        :param thread: The analysis thread where the exception occured, to be terminated.
        """
        dlg.close()
        thread.quit()
        dialog_utils.print_error(constants.UNEXPECTED_EXCEPTION_MSG)

    def wait_dlg_close_event(self, event, dlg, thread):
        """
        The function to be called when user clicks the "X" button on the analysis progress dialog.
        :param event: the close event came from the UI
        :param dlg: the dialog to be closed
        :param thread: the analysis thread to be terminated
        """
        thread.terminate() # TODO: terminate() is not recommended, but quit() doesn't work for some reason
        event.accept()

    def update_tasks(self):
        """
        Method for updating the UI when user changes the domain
        """
        self.ui.taskComboBox.clear()
        domain = constants.Domain[self.ui.domainComboBox.currentText()]
        self.ui.taskComboBox.addItems([task.name for task in domain.value])

    def onPredictedInputBrowseButtonClicked(self):
        """
        The function to be called when user clicks on the predicted "browse" button.
        """
        dir = definitions.DATA_DIR
        dialog_utils.browse_files(self.ui.selectPredictedLineEdit, dir)

    def onActualInputBrowseButtonClicked(self):
        """
        The function to be called when user clicks on the actuals "browse" button.
        """
        dir = definitions.ROOT_DIR
        dialog_utils.browse_files(self.ui.addActualLineEdit, dir)

    def onRunAnalysisButtonClicked(self):
        """
        The function to be called when user clicked on "Analyze" button.
        Starts the analysis progress according to the action selected in the UI.
        """
        predicted_files_str = self.ui.selectPredictedLineEdit.text()
        if not predicted_files_str:
            dialog_utils.print_error(constants.PROVIDE_INPUT_MSG)
            return

        self.task = constants.Task[self.ui.taskComboBox.currentText()]
        subjects = self.__create_subjects(self.task, predicted_files_str)

        if not subjects:
            return

        if len(subjects) > constants.MAX_SUBJECTS:
            dialog_utils.inform_user("Too many files to process. Maximum number is 25 files.")
            return

        # Check all input provided
        if not predicted_files_str:
            dialog_utils.print_error(constants.PROVIDE_INPUT_MSG)
            return

        # Prepare additional analysis parameters
        analysis_task = None
        outputdir = get_analysis_results_folder()
        other_path = path_utils.get_canonical_path(self.task)
        if self.ui.analysisMeanRadioButton.isChecked():
            analysis_task = AnalysisTask.Analysis_Mean

        elif self.ui.analysisCorrelationsRadioButton.isChecked():
            analysis_task = AnalysisTask.Analysis_Correlations
            # TODO: add other_path = ...

        else:
            dialog_utils.print_error(constants.SELECT_ACTION_MSG)
            return

        thread = AnalysisWorkingThread(analysis_task, subjects, self.task, outputdir, other_path)
        dlg = analysis_working_dlg_controller.AnalysisWorkingDlg()
        dlg.closeEvent = lambda event: self.wait_dlg_close_event(event, dlg, thread)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.show()
        thread.progress_finished_sig.connect(lambda: self.__handle_results(analysis_task, dlg, thread.results, subjects))
        thread.exception_occurred_sig.connect(lambda: self.__handle_unexpected_exception(dlg, thread))
        thread.start()

    def onRunComparisonButtonClicked(self):
        """
        The function to be called when user clicked on "Compare" button.
        Starts the analysis progress according to the action selected in the UI.
        """
        predicted_files_str = self.ui.selectPredictedLineEdit.text()
        actual_files_str = self.ui.addActualLineEdit.text()

        if not predicted_files_str or not actual_files_str:
            dialog_utils.print_error(constants.PROVIDE_INPUT_MSG)
            return

        self.task = constants.Task[self.ui.taskComboBox.currentText()]
        subjects = self.__create_subjects(self.task, predicted_files_str, actual_files_str)
        if not subjects:
            return

        if len(subjects) > constants.MAX_SUBJECTS:
            dialog_utils.inform_user("Too many files to process. Maximum number is 25 files.")
            return

        # Prepare additional analysis parameters
        analysis_task = None
        outputdir = get_analysis_results_folder()
        other_path = path_utils.get_canonical_path(self.task)

        if self.ui.comparisonCorrelationsRadioButton.isChecked():
            analysis_task = AnalysisTask.Compare_Correlations

        elif self.ui.comparisonSignificantRadioButton.isChecked():
            analysis_task = AnalysisTask.Compare_Significance

        else:
            dialog_utils.print_error(constants.SELECT_ACTION_MSG)
            return

        thread = AnalysisWorkingThread(analysis_task, subjects, self.task, outputdir, other_path)
        dlg = analysis_working_dlg_controller.AnalysisWorkingDlg()
        dlg.closeEvent = lambda event: self.wait_dlg_close_event(event, dlg, thread)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.show()
        thread.progress_finished_sig.connect(lambda: self.__handle_results(analysis_task, dlg, thread.results, subjects))
        thread.exception_occurred_sig.connect(lambda: self.__handle_unexpected_exception(dlg, thread))
        thread.start()
        # TODO: duplicated code with onAnalysisButtonClicked