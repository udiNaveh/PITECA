from sharedutils.linalg_utils import *
from sharedutils.subject import *
import numpy as np
import tensorflow as tf
from sharedutils.constants import *
from sharedutils.io_utils import *
import os.path
from abc import ABC, abstractmethod

ica_both_lowdim_path = os.path.join(PATHS.DATA_DIR, 'HCP_200', 'ica_both_lowdim.dtseries.nii')
betas_path = os.path.join(PATHS.DATA_DIR, 'regression_model','loo_betas_7_tasks', 'average_betas_100_subjects_7_tasks.npy')

temperature = 3.5

class IModel(ABC):

    def __init__(self, tasks):
        self.tasks = tasks
        super().__init__()

    @abstractmethod
    def predict(self, subject):
        pass


class LinearModel(IModel):

    available_tasks = {Task.MATH_STORY : 0,
                       Task.TOM: 1,
                       Task.MATCH_REL : 2,
                       Task.TWO_BK: 3,
                       Task.REWARD : 4,
                       Task.FACES_SHAPES : 5,
                       Task.T : 6
                       }

    def __init__(self, tasks):
        super(LinearModel, self).__init__(tasks)
        self.__feature_extractor = FeatureExtractor()
        self.__betas = None
        self.__soft_filters = None
        self.__is_loaded = False

    def __load(self):
        all_betas = np.load(betas_path)
        missing_tasks = []
        for task in self.tasks:
            if task not in LinearModel.available_tasks:
                missing_tasks.append(task)
                raise RuntimeWarning("no linear model for task {}".format(task.name))
        self.tasks = [t for t in self.tasks if t not in missing_tasks]
        tasks_indices = [LinearModel.available_tasks[t] for t in self.tasks]
        self.__betas = all_betas[tasks_indices,:,:]

        ica_both_lowdim, (series, bm) = cifti.read(ica_both_lowdim_path)
        self.__spatial_filters_soft = tempered_filters = softmax(np.transpose(ica_both_lowdim)* temperature)
        return True

    def __preprocess(self, subject_features):
        subject_features = np.concatenate((np.ones([STANDART_BM.N_CORTEX, 1]), np.transpose(subject_features)), axis=1)
        subject_features = demean_and_normalize(subject_features)
        subject_features[:, 0] = 1.0
        return subject_features

    def predict(self, subject):
        fe = self.__feature_extractor
        if not self.__is_loaded:
            self.__is_loaded = self.__load()
        betas = self.__betas
        prediction_paths = {}
        arr, bm = fe.get_features(subject)
        subject_feats = self.__preprocess(arr)
        dotprod = subject_feats.dot(betas)
        pred = np.sum(np.swapaxes(dotprod, 0, 1) * self.__spatial_filters_soft[:STANDART_BM.N_CORTEX,:], axis=2)
        for i,task in enumerate(self.tasks):
            predicted_task_activation = pred[i,:]
            save_to_dtseries(subject.predicted_task_filepath(task), bm, predicted_task_activation)
            prediction_paths[task] = save_to_dtseries(subject.predicted_task_filepath(task), bm, predicted_task_activation)

        return prediction_paths





class Model:

    def __init__(self, tasks):
        self.tasks = tasks
        self.is_loaded = False
        self.__graph__ = None

    def load(self):
        '''
        builds the computational graph (in tensorflow)
        The network is defined with a placeholder for input layer that can be assigned
        feature maps (numpy ndarray in size N_VERTICES_CORTEX * N_FEATURES).
        The output layer is a tensor of size  N_VERTICES_CORTEX * len(tasks).
        The model architecture, or at least its weights, is loaded from disc.
        I still havn't gone into tensor flow saving and loading of models so not really sure
        how to implement this.
        
         
        '''

        # do many things that
        self.__graph__ = "whatever" # todo
        self.is_loaded = True

    def preprocess(self, subject_features):
        '''
        prepare the features before running the network on them. notice that we can implement more sophisticated 
        processing here than just demean and normalize.
        :param subject_features: probably the functional connectivity features, as they appear in
        the .nii feature files.
        :return: the processed features. the output of this function is the input for the network 
        (i.e. for predict_all_tasks method)
        '''
        return demean_and_normalize(subject_features)

    def predict_all_tasks(self, subject_features):

        if not self.is_loaded:
            self.load()
        predicted_maps = np.zeros(STANDART_BM.N_CORTEX, len(self.tasks))
        subject_features = self.preprocess(subject_features)

        # get the prediction. for example something like:
        #with tf.Session as sess:
        #    predicted_maps = sess.run(output_layer, feed_dict= {subject_features = subject_features})

        return {task : predicted_maps[:,i] for i, task in enumerate(self.tasks)}

    def predict(self, subject):
        return


class FeatureExtractor:

    def __init__(self):
        self.matrices = dict()
        self.is_loaded = False

    def load(self):
        arr, _ = open_cifti(PATHS.SC_CLUSTERS)
        self.matrices['SC_CLUSTERS'] = arr
        arr, _ = open_cifti(PATHS.ICA_LR_MATCHED)
        self.matrices['ICA_LR_MATCHED'] = arr
        self.is_loaded = True
        return

    def get_features(self,subject):
        assert isinstance(subject,Subject)
        if subject.features_exist:
            arr, (series, bm) = open_cifti(subject.features_path)
            return arr, bm
        else:
            if not self.is_loaded:
                self.load()
            return self.extract_features(subject)

    def extract_features(self,subject):
        raise  NotImplementedError






