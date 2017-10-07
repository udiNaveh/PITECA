from sharedutils.constants import *
from sharedutils.general_utils import *
from os.path import join as join_path
from sharedutils import path_utils, dialog_utils, constants
import os


class Subject:

    def __init__(self, subject_id=None, output_dir=None, input_path=None, features_path=None, features_exist=False,
                 predicted={}, actual = {}):
        self.subject_id = subject_id
        self.output_dir = output_dir
        self.input_path = input_path
        self.features_exist = features_exist
        self.features_path = features_path
        self.predicted = predicted  # type: dict[Task, str]
        self.actual = actual  # type: dict[Task, str]

    def predicted_task_filepath(self, task):
        for d in Domain:
            if task in d.value:
                break
        return join_path(self.output_dir, "{0}_{1}_{2}_predicted".format(self.subject_id, d.name, task.name))

    def get_predicted_task_filepath(self, task):
        filename = path_utils.generate_file_name(self.output_dir, task, "{0}_{1}_predicted".format(self.subject_id, task.full_name))
        return path_utils.generate_final_filename(filename)

    def get_actual_task_filepath(self, task, actual_dir):
        filename = path_utils.generate_file_name(actual_dir, task, "{0}_{1}".format(self.subject_id, task.full_name))
        return path_utils.generate_final_filename(filename)

# TODO: make order in those functions and remove their documentation


'''
Function for Keren's needs
(prediction flow + analysis flow)
'''
def create_subjects(input_files_str, output_dir):
    errors = 0
    subjects = []
    input_files = path_utils.extract_filenames(input_files_str)
    for input_path in input_files:
        id = path_utils.get_id(input_path)
        if id is None:
            errors += 1
        else:
            features_path = path_utils.get_features_path(id)
            subject = Subject(id, output_dir, input_path, features_path)
            subjects.append(subject)
    if errors > 0:
        dialog_utils.inform_user(constants.NAMING_CONVENTION_ERROR)
        return []
    return subjects
