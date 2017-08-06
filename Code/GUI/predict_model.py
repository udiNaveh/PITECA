from PyQt5.QtCore import QFile, QIODevice
from sharedutils.constants import *
from sharedutils.path_utils import *
import os
from model.model import Model
from sharedutils.subject import Subject


class PredictModel:
    def __init__(self, input_files_str, output_dir, tasks):
        self.input_files = input_files_str[1:].split(',')
        self.output_dir = output_dir
        self.tasks = tasks


    def prepare_subjects(self):
        '''
        prepare subject objects whose prediction is desired, based on the processed input from the user
        in the Predict tab.
        :return: list of Subject
        '''
        subjects = []
        subjects_with_feats = [] # keep track on subjects that apparently already have existing features

        # set initial properties
        for file in self.input_files:
            # TODO: if we enable editing in the Line Edit (as now), we need to assert here that we have valid path
            subject = Subject()
            subject.id = get_id(file)
            subject.output_path = get_output_path(self.output_dir, subject.id)
            subject.input_path = file
            subject.features_path = get_features_path(subject.id)
            if os.path.isfile(subject.features_path):
                subjects_with_feats.append(subject)
            subjects.append(subject)

        # ask user permission to use existing features
        if subjects_with_feats:
            print(1)

    def predict(self):
        prediction_model = Model()
        subjects = self.prepare_subjects()
        for subject in subjects:
            prediction_model.predict(subject)


