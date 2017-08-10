from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
import sys, time, math
from GUI.thread import progressThread
from GUI.popups.progressBarView import Ui_ProgressBar
from model.models import Model
from sharedutils.constants import *
from sharedutils.dialog_utils import *
from sharedutils.path_utils import *
from sharedutils.subject import Subject


class PredictModel:
    def __init__(self, input_files_str, output_dir, tasks):
        self.input_files = input_files_str[1:].split(',') # TODO: handle edge cases
        self.output_dir = output_dir
        self.tasks = tasks
        self.subjects = []
        self.subjects_with_feats = []
        self.prediction_model = None

    def prepare_subjects(self):
        '''
        prepare subject objects whose prediction is desired, based on the processed input from the user
        in the Predict tab.
        :return: list of Subject
        '''

        # set initial properties
        for file in self.input_files:
            # TODO: if we enable editing in the Line Edit (as now), we need to assert here that we have valid path
            subject = Subject()
            subject.id = get_id(file)
            subject.output_path = get_output_path(self.output_dir, subject.id)
            subject.input_path = file
            subject.features_path = get_features_path(subject.id)
            if os.path.isfile(subject.features_path):
                self.subjects_with_feats.append(subject)
            self.subjects.append(subject)

        return self.subjects_with_feats

    def run_prediction_flow(self, ui):
        # Setup
        self.prediction_model = Model(self.tasks)
        self.prepare_subjects()

        # Open a dialog to ask the user for permission to use existing features
        if self.subjects_with_feats:
            use_existing = ask_user(False, QUESTION_TITLE, EXIST_FEATS_MSG)

        # Update subjects according to user answer
        if use_existing:
            for subject in self.subjects_with_feats:
                subject.features_exist = True

        # Start prediction process

        # prepare progress bar dialog
        progress_bar_dlg = QtWidgets.QDialog()
        progress_bar_dlg.setWindowModality(Qt.ApplicationModal)
        self.progress_bar_ui = Ui_ProgressBar()
        self.progress_bar_ui.setupUi(progress_bar_dlg)
        # start a new thread to do the prediction
        self.progress_thread = progressThread(self.prediction_model, self.subjects, self.progress_bar_ui.progressBar)
        # setup cancel button functionality
        self.progress_bar_ui.pushButton.clicked.connect(lambda: self.progress_thread.terminate())

        # start work
        progress_bar_dlg.show()
        self.progress_thread.start()
        progress_bar_dlg.exec_()
