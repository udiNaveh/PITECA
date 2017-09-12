from sharedutils.linalg_utils import *
from sharedutils.subject import *
import numpy as np
import tensorflow as tf
from sharedutils.constants import *
from sharedutils.io_utils import *
import os.path
import definitions
from abc import ABC, abstractmethod
import definitions
import time
from scipy import sparse

# constants - use configs instead

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
    # a mapping of currently available tasks in the model to their ordinal number in the
    # data files related to the tasks. e.g. in the betas matrix the betas related to Task.REWARD
    # are in the coordintas [4,:,:]

    def __init__(self, tasks):
        super(LinearModel, self).__init__(tasks)
        self.__feature_extractor = FeatureExtractor()
        self.__betas = None
        self.spatial_filters_soft = None
        self.__is_loaded = False

    def __load(self):
        '''
        Loads from disk all the data matrices needed for the model. This is done once per
        instance of LinearModel.
        :return: 
        '''
        all_betas = np.load(definitions.LINEAR_MODEL_BETAS_PATH)
        missing_tasks = []
        for task in self.tasks:
            if task not in LinearModel.available_tasks:
                missing_tasks.append(task)
                raise RuntimeWarning("no linear model for task {}".format(task.name))
            # @error_handle. Notice that this is probably only for development,
            # as the release version will include anyway only the tasks
            # for which we have the model.

        self.tasks = [t for t in self.tasks if t not in missing_tasks]
        tasks_indices = [LinearModel.available_tasks[t] for t in self.tasks]
        self.__betas = all_betas[tasks_indices,:,:]

        ica_both_lowdim, (series, bm) = cifti.read(definitions.ICA_LOW_DIM_PATH)
        self.spatial_filters_soft = softmax(np.transpose(ica_both_lowdim) * temperature)
        below_threshold = self.spatial_filters_soft < 1e-2
        self.spatial_filters_soft[below_threshold] = 0
        self.spatial_filters_soft = self.spatial_filters_soft[:STANDART_BM.N_CORTEX, :]
        self.__spatial_filters_hard = np.argmax(np.transpose(ica_both_lowdim[:, :STANDART_BM.N_CORTEX]), axis = 1)
        return True

    def __preprocess(self, subject_features):
        subject_features = np.concatenate((np.ones([STANDART_BM.N_CORTEX, 1]), np.transpose(subject_features)), axis=1)
        subject_features = demean_and_normalize(subject_features)
        subject_features[:, 0] = 1.0
        return subject_features

    def predict(self, subject, filters = 'soft', save = True):
        fe = self.__feature_extractor
        if not self.__is_loaded:
            self.__is_loaded = self.__load()
        betas = self.__betas
        prediction_paths = {}
        arr, bm = fe.get_features(subject)
        subject_feats = self.__preprocess(arr)
        if filters == 'soft':
            start = time.time()
            dotprod = subject_feats.dot(betas)
            pred = np.sum(np.swapaxes(dotprod, 0, 1) * self.spatial_filters_soft, axis=2)
            stop = time.time()
            print("soft filter prediction took {0:.4f} seconds".format(stop-start))
        elif filters == 'hard':
            start = time.time()
            pred = np.zeros([STANDART_BM.N_CORTEX, len(self.tasks)])
            for j in range(np.size(self.spatial_filters_soft, axis=1)):
                ind = self.__spatial_filters_hard == j
                M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
                                    demean(subject_feats[ind, 1:])),axis=1)
                pred[ind, :] = M.dot((betas[:,:,j]).swapaxes(0,1))
            stop = time.time()
            print("hard filter prediction took {0:.4f} seconds".format(stop - start))
        elif filters == 'soft fast':
            start = time.time()
            pred = np.zeros([STANDART_BM.N_CORTEX, len(self.tasks)])
            for j in range(np.size(self.spatial_filters_soft, axis=1)):
                ind = self.__spatial_filters_hard == j
                M = np.concatenate((subject_feats[ind,0].reshape(np.count_nonzero(ind),1),\
                                    demean(subject_feats[ind, 1:])),axis=1)
                pred[ind, :] = M.dot((betas[:,:,j]).swapaxes(0,1))
            stop = time.time()
            print("soft fast filter prediction took {0:.4f} seconds".format(stop - start))
        if not save:
            return pred

        for i,task in enumerate(self.tasks):
            predicted_task_activation = pred[i,:]
            if save:
                prediction_paths[task] = save_to_dtseries(subject.get_predicted_task_filepath(task), bm, predicted_task_activation)
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
        arr = np.load(definitions.SC_CLUSTERS_PATH)
        self.matrices['SC_CLUSTERS'] = arr
        arr = np.load(definitions.ICA_LR_MATCHED_PINV_PATH)
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
        rfmri_data, (series, bm) = open_cifti(subject.input_path) # arr is time * 91k
        n_vertices, t = rfmri_data.shape

        # preprocess
        rfmri_data_normalized = variance_normalise(rfmri_data) # % noise variance normalisation
        data = detrend(rfmri_data_normalized) # subtract linear trend from time series each vertex

        # perform dual regression to obtain individual cortical spatial maps
        # step 1 - get subject's individual time series for each by network
        ts_by_network = np.dot(data, self.matrices['ICA_LR_MATCHED'])  # time x 76
        # step 2 - get subject's individual cortical spatial maps
        LHRH = fsl_glm(ts_by_network, data).transpose() # 91k x 76

        # create spatial maps for the whole brain
        ROIS = np.zeros([STANDART_BM.N_TOTAL_VERTICES, 108])
        #  left hemisphere networks
        ROIS[: STANDART_BM.N_LH, : 38] = LHRH[: STANDART_BM.N_LH, : 38]
        # right hemisphere networks
        ROIS[STANDART_BM.N_LH : STANDART_BM.N_CORTEX , 38 :76] = \
            LHRH[STANDART_BM.N_LH : STANDART_BM.N_CORTEX , 38 :76]
        # for subcortical networks - use data of group
        ROIS[:, 76:] = self.matrices['SC_CLUSTERS']
        rfmri_data_normalized = demean(rfmri_data_normalized)  # remove mean from each column
        # multiple regression - characteristic time series for each network
        T2 = np.dot(np.linalg.pinv(ROIS), rfmri_data_normalized.transpose()) # 108 x time
        # get the featires - correlation coefficient for each vertex with each netwrok
        features_map = np.dot(normalize(T2, axis=1), normalize(rfmri_data_normalized, axis=0))
        save_to_dtseries(subject.features_path, bm, features_map)
        return features_map












