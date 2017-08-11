from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
import sys, time, math
from GUI.progress_thread import ProgressThread
from GUI.popups.progressBarView import Ui_ProgressBar
from model.models import Model
from sharedutils.constants import *
from sharedutils.dialog_utils import *
from sharedutils.path_utils import *
from sharedutils.subject import Subject
import os.path


def _extract_filenames(input_files_str):
    pathes = []
    filenames = input_files_str[1:-1].split(', ')
    for filename in filenames:
        start = filename.index("'")
        end = filename.rindex("'")
        pathes.append(filename[start+1:end])
    return pathes


class PredictTabModel:
    def __init__(self, input_files_str, output_dir, tasks):
        self.input_files = _extract_filenames(input_files_str)
        self.output_dir = output_dir
        self.tasks = tasks
        self.subjects = []
        self.subjects_with_feats = []
        self.prediction_model = None

    def _prepare_subjects(self):
        '''
        prepare subject objects whose prediction is desired, based on the processed input from the user
        in the Predict tab. In addition, prepare a list with subjects that already seem to have features.
        '''
        # set initial properties
        for file in self.input_files:
            # TODO: if we enable editing in the Line Edit, we need to assert here that we have valid path
            subject = Subject()
            subject.id = get_id(file)
            subject.output_path = get_output_path(self.output_dir, subject.id)
            subject.input_path = file
            subject.features_path = get_features_path(subject.id)
            self.subjects.append(subject)
            if path.exists(subject.features_path):
                self.subjects_with_feats.append(subject)

    def run_prediction_flow(self, ui):
        # Setup
        self.prediction_model = Model(self.tasks)
        self._prepare_subjects()

        # Open a dialog to ask the user for permission to use existing features
        use_existing = False
        if self.subjects_with_feats:
            use_existing = ask_user(use_existing, QUESTION_TITLE, EXIST_FEATS_MSG)
        # TODO: handle situations when user press X button
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
        self.progress_thread = ProgressThread(self.prediction_model, self.subjects, self.progress_bar_ui.progressBar)
        # setup cancel button functionality
        self.progress_bar_ui.pushButton.clicked.connect(lambda: self.progress_thread.terminate()) # TODO: terminate() is not recommended
        # setup exit functionality
        self.progress_bar_ui

        # TODO: terminate working thread when dialog closed
        # TODO: change button text to 'Finished' or disable it when prediction completed

        # start work
        progress_bar_dlg.show()
        self.progress_thread.start()
        progress_bar_dlg.exec_()
