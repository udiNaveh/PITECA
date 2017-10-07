from PyQt5.QtCore import Qt

from GUI.popups.predict_working_dlg_controller import PredictWorkingDlg
from model.models import model_factory, IModel
from sharedutils.constants import *
from sharedutils.dialog_utils import *
from sharedutils.subject import create_subjects
from GUI import settings_controller


class PredictTabModel:
    def __init__(self, input_files_str, output_dir, tasks):
        self.input_files_srt = input_files_str
        self.output_dir = output_dir
        self.tasks = tasks
        self.subjects = []
        self.subjects_with_feats = []
        self.prediction_model = None
        self.progress_thread = None

    def __prepare_tasks(self):
        res = []
        for task in self.tasks.values():
            for contrast in task:
                res.append(contrast)
        return res


    def __prepare_subjects(self):
        '''
        prepare subject objects whose prediction is desired, based on the processed input from the user
        in the Predict tab. In addition, prepare a list with subjects that already seem to have features.
        '''
        self.subjects = create_subjects(self.input_files_srt, self.output_dir)
        for subject in self.subjects:
            # TODO: if we enable editing in the Line Edit, we need to assert that we have valid paths
            if path.exists(subject.features_path):
                self.subjects_with_feats.append(subject)


    def __handle_predict_exception(self, progress_bar_dlg):
        progress_bar_dlg.progress_thread.quit()
        progress_bar_dlg.close()
        print_error(UNEXPECTED_EXCEPTION_MSG)

    def run_prediction_flow(self, ui):
        # Setup
        MODEL_NAME = settings_controller.get_model()
        self.prediction_model = model_factory(MODEL_NAME, self.tasks)
        self.__prepare_subjects()
        if len(self.subjects) > MAX_SUBJECTS:
            inform_user("Too many files to process. Maximum number is 25 files.")
            return
        if len(self.subjects) == 0:
            return

        # Open a dialog to ask the user for permission to use existing features
        use_existing = False
        if self.subjects_with_feats:
            use_existing = ask_user(use_existing, QUESTION_TITLE, EXIST_FEATS_MSG)
        # TODO: handle situations when user press X button
        # Update subjects according to user answer
        if use_existing:
            for subject in self.subjects_with_feats:
                subject.features_exist = True

        # prepare progress bar dialog
        progress_bar_dlg = PredictWorkingDlg(self.prediction_model, self.subjects)
        progress_bar_dlg.progress_thread.exception_occurred_sig.connect(
            lambda: self.__handle_predict_exception(progress_bar_dlg)
        )
        progress_bar_dlg.setWindowModality(Qt.ApplicationModal)

        # TODO: change button text to 'Finished' or disable it when prediction completed

        # start work
        progress_bar_dlg.show()
        progress_bar_dlg.start_progress()
        progress_bar_dlg.exec_()


