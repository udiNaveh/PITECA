from PyQt5.QtCore import QFile, QIODevice
from sharedutils.constants import *
from sharedutils.string_utils import *
import os
from model.model import Model
from sharedutils.subject import Subject

class PredictModel:
    def __init__(self, input_files_str, output_dir, tasks):
        self.input_files = input_files_str[1:].split(',')
        self.output_dir = output_dir
        self.tasks = tasks

    def find_existing_features(self):
        f = QFile(CONFIG_PATH)
        f.open(QIODevice.ReadOnly)
        features_dir = f.readAll()
        f.close()

        exist_feats_input_files = []
        for file_name in self.input_files:
            id = get_id(file_name)
            path = get_fetures_path(id)
            if os.path.isfile(path):
                exist_feats_input_files.append(file_name)

        return exist_feats_input_files

    def extract_features(self, existing_features):
        for file in self.input_files:
            if file not in existing_features:
                print(file)


    def prepare_subjects(self):
        '''
        prepare subject objects whose prediction is desired, based on the processed input from the user
        in the Predict tab, stored in this (self) PredictModel object.
        :return: list of Subject
        '''
        subjects = []
        for file_path in self.input_files:

            subject = Subject()
            subject.id = get_id(file_path)
            subject.output_path = get_full_path(self.output_dir, subject.id)
            subject.input_path = file_path
            subject.features_path = get_fetures_path(subject.id)



            return


    def predict(self):
        prediction_model = Model()
        subjects = self.prepare_subjects()
        for subject in subjects:
            prediction_model.predict(subject)


