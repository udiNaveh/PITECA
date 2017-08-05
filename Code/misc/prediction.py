import numpy as np
import os
from constants import *
import cifti
from linalg_utils import *
from ml_utils import *
from io_utils import *
from model import *
from asynch_utils import *

'''
This module handles anything that is related to the predict module/tab in PITECA,
except for the actual implementation of the model.



'''


class Subject:
    '''
    This doesn't really belong here. just didn;t find another place. Also, there is no justification to make Subject a class if it
    only has 3 fields and no methods. Need to think about it.
    '''

    def __init__(self, input_dtseires, feature_path, output_files):
        self.input_dtseires = input_dtseires
        self.feature_path = feature_path
        self.output_files = output_files


####

def run_all_predictions(tasks, subjects, configs):
    '''
    this method runs the main session of a prediction. It is called after the user chooses
    the input rfmri files and output task + some basic checks were run on the chosen files (e.g. naming conventions)
    and Subjects were initialized.
    This method has to support multi-threading/ async programming, as it can take a lot to execute,
    and it can also trigger UI events and wait for response (for example when features file for a subject already) 
    exists and need to ask whether to use it, to extract features and replace existing file, or extract or to extract features and
    keep both files).
    
    :param tasks: a list of tasks to predict (represented as strings? maybe enums?)  
    :param subjects: a list of Subjects.
    :param configs: some user defined configurations that can be relevant here. dor example, wether to
    save extracted features at all, or how to save 
    :return: 
    '''
    pass

    model = Model(tasks)
    for subject in subjects:
        use_existing = False
        if os.path.isfile(subject.feature_path):
            use_existing, subject.feature_path = do_use_existing_features(subject.feature_path)

        if use_existing:
            subject_features, (series, bm) = open_features_file(subject.feature_path)
        else:
            arr, (series, bm) = open_rfmri_file(subject.input_dtseires)
            arr = preprocess_rfmri_data(arr)
            subject_features = extract_features(arr)
            if os.path.exists(os.path.dirname(subject.feature_path)):
                save_to_dtseries(subject.feature_path, bm, subject_features)

        subject_predictions = model.predict_all_tasks(subject_features)

        # The way prediction files are saved can be configured in different ways,
        # Ido said that we should always save dtseries, but at least with the cifti python package
        # it was rather easy to save other types. So maybe we want to enable the users to choose for themselves,
        # and then we can have multiple saving options here (for example we can save all tasks for same subject)
        # in one dscalar file, where each task has its name.

        for task in subject_predictions:
            try:
                save_to_dtseries(subject.output_files[task])
            except Exception as e:
                pass
                # we actually need to handle possible exceptions here and in other places.
                # for example, we may write it to the logger.


def extract_features(arr):
    '''
    Havn't implemented it yet. Will be based on Ido's code, though we might want to change some bits.
    :param arr: n_time_units x N_VERTICES matrix with a subject's remri data seires, after preprocessing.
    :return: the extracted features matrix.
    '''
    return None


def preprocess_rfmri_data(data):
    return detrend(variance_normalise(data))



