from sharedutils.constants import *
from os.path import join as join_path
from sharedutils import path_utils, dialog_utils, constants


class Subject:
    """
    A class to represent a subject element that contains relevant data about the subjects in the prediction
    and analysis flows, across modules.
    """

    def __init__(self, subject_id=None, output_dir=None, input_path=None, features_path=None, features_exist=False,
                 predicted={}, actual = {}):
        self.subject_id = subject_id
        self.output_dir = output_dir
        self.input_path = input_path
        self.features_exist = features_exist
        self.features_path = features_path
        self.predicted = predicted  # type: dict[Task, str]
        self.actual = actual  # type: dict[Task, str]

    def get_predicted_task_filepath(self, task):
        filename = path_utils.generate_file_name(self.output_dir, task, "{0}_{1}_predicted".format(self.subject_id, task.full_name))
        return path_utils.generate_final_filename(filename)


def create_subjects(input_files_str, output_dir):
    """
    A utility function to prepare Subject elements in the beginning of the prediction flow,
    with data from the UI.
    :param input_files_str: the string of all input files as got in the line edit ("[filepath1, ..., filepath23]")
    :param output_dir: the string of the output dir.
    :return: a list of Subject(s)
    """
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
