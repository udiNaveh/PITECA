from sharedutils.linalg_utils import *
import numpy as np
import tensorflow as tf
from sharedutils.constants import *


'''
Ideally, all the tensorflow code , and other code related to the prediction model
and to learning it will be in this file.
'''


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

