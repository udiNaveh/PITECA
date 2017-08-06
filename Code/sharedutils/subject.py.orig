from sharedutils.constants import *
from sharedutils.string_utils import *
from os.path import join as join_path


class Subject:

    def __init__(self, subject_id=None, output_path=None, input_path=None, features_exist=False, features_path=None,
                 predicted={}, actual = {}):
        self.subject_id = subject_id
        self.output_path = output_path
        self.input_path = input_path
        self.features_exist = features_exist
        self.features_path = features_path
        self.predicted = predicted # type: dict[Task, str]
        self.actual = actual # type: dict[Task, str]

    def predicted_task_filepath(self, task):
        for d in Domain:
            if task in d.value:
                break
        return join_path(self.output_path, "{0}_{1}_{2}_predicted".format(self.subject_id,d.name, task.name))



def create_subjects(ids, extracted_featuresr_dir, outputpath):
    subjects = []
    for id in ids:
        id = zeropad(id, 6)
        subjects.append(Subject(subject_id=id,
                                features_path=join_path(extracted_featuresr_dir, id + '_features.dtseries.nii'),
                                features_exist=True,
                                output_path = outputpath))
    return subjects