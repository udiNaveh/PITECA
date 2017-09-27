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
from misc.model_hyperparams import *
from misc.nn_model import *
import pickle
# constants - use configs instead






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

    def __init__(self, tasks, features = None):
        super(LinearModel, self).__init__(tasks)

        self.__betas = None
        self.spatial_filters_soft = None
        self.__is_loaded = False
        self.__feature_extractor = FeatureExtractor()

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
        self.spatial_filters_soft = softmax(np.transpose(ica_both_lowdim) * TEMPERATURE)
        below_threshold = self.spatial_filters_soft < 1e-2
        self.spatial_filters_soft[below_threshold] = 0
        self.spatial_filters_soft = self.spatial_filters_soft[:STANDART_BM.N_CORTEX, :]
        self.__spatial_filters_hard = np.argmax(np.transpose(ica_both_lowdim[:, :STANDART_BM.N_CORTEX]), axis = 1)
        return True

    def __preprocess(self, subject_features):
        subject_features = subject_features[:, :STANDART_BM.N_CORTEX]
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


class TFRoiBasedModel(IModel):

    available_tasks = {Task.MATH_STORY : 0,
                       Task.TOM: 1,
                       Task.MATCH_REL : 2,
                       Task.TWO_BK: 3,
                       Task.REWARD : 4,
                       Task.FACES_SHAPES : 5,
                       Task.T : 6
                       }

    def __init__(self, tasks):
        super(TFRoiBasedModel, self).__init__(tasks)
        self._weights = {}
        self.feature_extractor = FeatureExtractor()
        self.is_loaded =  False


        missing_tasks = []
        for task in self.tasks:
            if task not in TFRoiBasedModel.available_tasks:
                missing_tasks.append(task)
                print("no TFRoiBasedModel model for task {}".format(task.name))

        self.tasks = [t for t in self.tasks if t not in missing_tasks]

    def _load(self):
        spatial_filters_raw, (series, bm) = cifti.read(definitions.ICA_LOW_DIM_PATH)
        spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDART_BM.CORTEX])
        soft_filters = softmax(spatial_filters_raw.astype(float) * TEMPERATURE)
        soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
        soft_filters[:, 2] = 0
        soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDART_BM.N_CORTEX, 1])
        hard_filters = np.round(softmax(spatial_filters_raw.astype(float) * 1000))
        hard_filters[spatial_filters_raw < SPATIAL_FILTERS_THRESHOLD] = 0
        self.spatial_filters_soft = soft_filters
        self.spatial_filters_hard = hard_filters
        self.x, self.y_pred = self.get_placeholders()
        self.variables = self.get_trainable_variables()
        self.load_weights()
        self.spatial_filters_raw = spatial_filters_raw
        return True

    @abstractmethod
    def load_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def get_placeholders(self):
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_variables(self):
        raise NotImplementedError()

    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = demean_and_normalize(subject_features)
        return subject_features

    def predict(self, subject, save = True):
        if not self.is_loaded:
            self.is_loaded = self._load()
        arr, bm = self.feature_extractor.get_features(subject)
        subject_feats = self.preprocess_features(arr)
        subject_predictions = {}
        prediction_paths = {}
        with tf.Session() as sess:
            for task in self.tasks:
                subject_task_prediction = np.zeros([1, STANDART_BM.N_CORTEX])
                start = time.time()
                for j in range(np.size(self.spatial_filters_soft, axis=1)):
                    if j in self._weights[task]:
                        ind = self.spatial_filters_soft[: STANDART_BM.N_CORTEX, j] > 0
                        weighting = self.spatial_filters_soft[:,j][ind]
                        features = subject_feats[ind]
                        weights = self._weights[task][j]
                        assert len(self.variables) == len(weights) # TODO delete
                        region_feed_dict = union_dicts({self.x: features},
                                                       {tensor: weights[i] for i, tensor in enumerate(self.variables)})
                        roi_prediction = np.squeeze(sess.run(self.y_pred, feed_dict=region_feed_dict))
                        subject_task_prediction[:,ind] += weighting * roi_prediction
                subject_predictions[task] = subject_task_prediction
                end = time.time()
                print("preiction of task {0} subject {1} took {2:.3f}seconds".format(task.full_name, subject.subject_id, end-start))
                if save:
                    prediction_paths[task] = save_to_dtseries(subject.get_predicted_task_filepath(task), bm,
                                                              subject_task_prediction)
        return subject_predictions, prediction_paths


class NN2lhModel(TFRoiBasedModel):

    hl1_size = 50
    hl2_size = 50
    input_size = NUM_FEATURES

    def __init__(self, tasks):
        super(NN2lhModel, self).__init__(tasks)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.NN_WEIGHTS_DIR, '2hl_70s',
                           'nn_2hl_no_roi_normalization_70s_weights_{0}_all_filters.pkl'.format(task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
        regression_with_two_hidden_layers_build(input_dim= NN2lhModel.input_size, output_dim=1, scope_name='nn1_h2_reg',
                                                layer1_size=NN2lhModel.hl1_size, layer2_size=NN2lhModel.l2_size)
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nn1_h2_reg')]


class NN2lhModelWithFiltersAsFeatures(TFRoiBasedModel):

    hl1_size = 50
    hl2_size = 50
    input_size = NUM_FEATURES + NUM_SPATIAL_FILTERS

    def __init__(self, tasks):
        super(NN2lhModelWithFiltersAsFeatures, self).__init__(tasks)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.NN_WEIGHTS_DIR, '2hl_features_as_filters_70s',
                           'nn_2hl_no_roi_normalization_fsf_70s_weights_{0}_all_filters.pkl'.format(task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
        regression_with_two_hidden_layers_build(input_dim= NN2lhModelWithFiltersAsFeatures.input_size, output_dim=1, scope_name='nn1_h2_reg_fsf',
                                                layer1_size=NN2lhModelWithFiltersAsFeatures.hl1_size, layer2_size=NN2lhModelWithFiltersAsFeatures.hl2_size)
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nn1_h2_reg_fsf')]

    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = demean_and_normalize(subject_features)
        subject_features = np.concatenate((subject_features, self.spatial_filters_raw), axis = 1)
        return subject_features



class TFLinear(TFRoiBasedModel):
    def __init__(self, tasks):
        super(TFLinear, self).__init__(tasks)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'learned_by_roi_70s',
                                        'linear_weights_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))

            self._weights[task] = {i : [weights[:,i:i+1]] for i in range(np.size(weights, axis=1)) if i!=2}

    def get_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES+1, output_dim=1, scope_name='lin_reg')
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lin_reg')][:1]

    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = demean_and_normalize(subject_features)
        return add_ones_column(subject_features)


class TFLinearAveraged(TFRoiBasedModel):
    def __init__(self, tasks):
        super(TFLinearAveraged, self).__init__(tasks)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'averaged_weights_70s',
                                        'linear_weights_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))

            self._weights[task] = {i : [weights[:,i:i+1]] for i in range(np.size(weights, axis=1)) if i!=2}

    def get_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES+1, output_dim=1, scope_name='lin_reg')
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lin_reg')][:1]

    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = demean_and_normalize(subject_features)
        return add_ones_column(subject_features)


class TFLinearFSF(TFRoiBasedModel):
    input_size = NUM_FEATURES + NUM_SPATIAL_FILTERS

    def __init__(self, tasks):
        super(TFLinearFSF, self).__init__(tasks)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'learned_by_roi_filters_as_features70s',
                                        'linear_weights_fsf_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))

            self._weights[task] = {i : [weights[:,i:i+1]] for i in range(np.size(weights, axis=1)) if i!=2}

    def get_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES+1+ NUM_SPATIAL_FILTERS, output_dim=1, scope_name='lin_reg')
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lin_reg')][:1]


    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = demean_and_normalize(subject_features)
        subject_features = np.concatenate((subject_features, self.spatial_filters_raw), axis = 1)
        return add_ones_column(subject_features)

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
        return features_map, bm


available_models = {
    'Linear by ROI averaged betas': TFLinearAveraged,
    'Linear by ROI': TFLinear,
    'Linear by ROI with group connectivity features': TFLinearFSF,
    'MLP by ROI' : NN2lhModel,
    'MLP by ROI with group connectivity features' : NN2lhModelWithFiltersAsFeatures
}

def model_factory(model_name, tasks):
    if model_name not in available_models:
        raise ValueError("There is no prediction model named {}".format(model_name))
    my_model_class = available_models[model_name]
    my_model = my_model_class(tasks)
    return my_model







