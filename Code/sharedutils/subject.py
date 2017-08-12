from sharedutils.constants import *
from sharedutils.general_utils import *
from os.path import join as join_path
from sharedutils import path_utils


class Subject:

    def __init__(self, subject_id=None, output_path=None, input_path=None, features_path=None, features_exist=False,
                 predicted={}, actual = {}):
        self.subject_id = subject_id
        self.output_path = output_path
        self.input_path = input_path
        self.features_exist = features_exist
        self.features_path = features_path
        self.predicted = predicted  # type: dict[Task, str]
        self.actual = actual  # type: dict[Task, str]

    def predicted_task_filepath(self, task):
        for d in Domain:
            if task in d.value:
                break
        return join_path(self.output_path, "{0}_{1}_{2}_predicted".format(self.subject_id,d.name, task.name))

    def get_predicted_task_filepath(self, task):
        return join_path(self.output_path, task.domain, task.name,
                         "{0}_{1}_predicted".format(self.subject_id, task.full_name))

    def get_predicted_actual_overlap_task_filepath(self, task, outputdir):
        return join_path(outputdir, "{0}_{1}_predicted_actual_overlap".format(self.subject_id,task.full_name))


# TODO: make order in those functions and remove their documentation

'''
Functions for Udi's needs
'''
def create_subjects(ids, extracted_featuresr_dir, outputpath):
    subjects = []
    for id in ids:
        id = zeropad(id, 6)
        subjects.append(Subject(subject_id=id,
                                features_path=join_path(extracted_featuresr_dir, id + '_features.dtseries.nii'),
                                features_exist=True,
                                output_path = outputpath))
    return subjects

'''
Function for Keren's needs
(prediction flow + analysis flow)
'''
# TODO: move this function to predict_model, because this is the only place where it is used
def create_subjects(input_files_str, output_dir):
    subjects = []
    input_files = path_utils.extract_filenames(input_files_str)
    for input_path in input_files:
        id = path_utils.get_id(input_path)
        output_path = path_utils.get_output_path(output_dir, id)
        features_path = path_utils.get_features_path(id)
        subject = Subject(id, output_path, input_path, features_path)
        subjects.append(subject)
    return subjects
