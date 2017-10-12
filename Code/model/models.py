"""
This module contains the different model that can be used
to predict and generate activation maps for subjects based on their
input rfmri data. 

"""
import os.path
import pickle
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import time
from sharedutils.constants import *
from sharedutils.io_utils import *
from sharedutils.linalg_utils import *
from sharedutils.general_utils import *
from sharedutils.subject import *
import definitions
from model.model_hyperparams import *
from model.data_manager import *
from model.nn_model import *






class IModel(ABC):
    """
    Base abstract class for all models
    """

    def __init__(self, tasks):
        self.tasks = tasks
        super().__init__()

    @abstractmethod
    def predict(self, subject):
        pass


class TFRoiBasedModel(IModel):
    """
    Base abstract class for all roi-based models. These models divide the cortex vertices into 49
    rois, where each roi has its own predictive model. The predicted value for each vertex is
    determined using the model of the roi to which it belongs. Note however that as soft filters are used,
    a vertex's relation to the different roi's is describes as a (usually sparse) probability vector.
    if vertex v is belongs 0.7 to roi j1 and 0.3 to roi j2, its predicted activation value will be
    0.7 * f_j1(x)... 
    
    """

    # TODO COMPLETE DOCUMENTATION


    def __init__(self, tasks, feature_extractor = None):
        super(TFRoiBasedModel, self).__init__(tasks)
        self._weights = {}
        self.feature_extractor = FeatureExtractor() if feature_extractor is None else feature_extractor
        self.is_loaded =  False
        self.scope_name = ''



    def _load(self):
        spatial_filters_raw, (series, bm) = cifti.read(definitions.ICA_LOW_DIM_PATH)
        spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDARD_BM.CORTEX])
        soft_filters = softmax(spatial_filters_raw.astype(float) * TEMPERATURE)
        soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
        soft_filters[:, 2] = 0
        soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDARD_BM.N_CORTEX, 1])
        hard_filters = np.round(softmax(spatial_filters_raw.astype(float) * 1000))
        hard_filters[spatial_filters_raw < SPATIAL_FILTERS_THRESHOLD] = 0
        # self.tasks_data = open_pickle(definitions.TASKS_CANONICAL_DATA2)
        self.spatial_filters = soft_filters
        self.spatial_filters_hard = hard_filters
        self.x, self.y_pred = self.get_placeholders()
        self.variables = self.get_trainable_variables()
        self.load_weights()
        self.spatial_filters_raw = spatial_filters_raw
        self.global_features = None
        return True

    @abstractmethod
    def load_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def get_placeholders(self):
        raise NotImplementedError()

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)]


    def update_feats(self, subject_feats, task):
        return subject_feats

    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = demean_and_normalize(subject_features)
        if self.global_features is not None:
            subject_features = np.concatenate([subject_features, self.global_features], axis=1)
        return subject_features

    def get_spatial_filters(self, subject_feats):
        return self.spatial_filters

    def predict(self, subject, save = True):
        """
        :param subject: an object of type Subject 
        :param save: bool - whether or not to save the predicted activation into .dtseries.nii files
        :return: a Tuple(subject_predictions, prediction_paths), with
                subject_predictions : a dictionary mapping tasks to prediction matrices
                prediction_paths : a dictionary mapping tasks to paths of saved predictions in .dtseries.nii files
        """
        if not self.is_loaded:
            self.is_loaded = self._load()
        arr, bm = self.feature_extractor.get_features(subject)
        subject_feats = self.preprocess_features(arr)
        spatial_filters = self.get_spatial_filters(subject_feats)
        subject_predictions, prediction_paths = {}, {}
        with tf.Session() as sess:
            for task in self.tasks:
                subject_feats = self.update_feats(subject_feats, task)
                subject_task_prediction = np.zeros([1, STANDARD_BM.N_CORTEX])
                start = time.time()
                for j in range(np.size(spatial_filters, axis=1)):
                    if j in self._weights[task]:
                        ind = spatial_filters[: STANDARD_BM.N_CORTEX, j] > 0
                        weighting = spatial_filters[:, j][ind]
                        features = subject_feats[ind]
                        weights = self._weights[task][j]
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

    def __init__(self, tasks, feature_extractor=None):
        super(NN2lhModel, self).__init__(tasks, feature_extractor)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.NN_WEIGHTS_DIR, '2hl_70s',
                           'nn_2hl_no_roi_normalization_70s_weights_{0}_all_filters.pkl'.format(task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
            regression_with_two_hidden_layers_build(input_dim= NN2lhModel.input_size, output_dim=1, scope_name='nn1_h2_reg',
                                                layer1_size=NN2lhModel.hl1_size, layer2_size=NN2lhModel.hl2_size)
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nn1_h2_reg')]


class NN2lhModelWithFiltersAsFeatures(TFRoiBasedModel):

    hl1_size = 50
    hl2_size = 50
    input_size = NUM_FEATURES + NUM_SPATIAL_FILTERS

    def __init__(self, tasks, feature_extractor=None):
        super(NN2lhModelWithFiltersAsFeatures, self).__init__(tasks, feature_extractor)

    def _load(self):
        super(NN2lhModelWithFiltersAsFeatures, self)._load()
        self.global_features = demean_and_normalize(self.spatial_filters_raw)
        return True

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



class NN2lhModelWithFiltersAndTaskAsFeatures(TFRoiBasedModel):



    def __init__(self, tasks, feature_extractor=None):
        super(NN2lhModelWithFiltersAndTaskAsFeatures, self).__init__(tasks, feature_extractor)
        self.scope_name = 'nn1_h2_reg_fsf_ms'

    def _load(self):
        super(NN2lhModelWithFiltersAndTaskAsFeatures, self)._load()
        self.global_features = demean_and_normalize(self.spatial_filters_raw)
        return True

    def update_feats(self, subject_feats, task):
        feats = np.concatenate([subject_feats[:,:158]] + [t[task] for t in self.tasks_data], axis=1)
        return feats

    def load_weights(self):
        for task in self.tasks:

            weights_path = os.path.join(definitions.LOCAL_DATA_DIR, 'model', 'nn', "nn_2hl_by_roi_fsf_ms__100s_weights_{0}_all_filters.pkl".
                         format(task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
            regression_with_two_hidden_layers_build(input_dim= 160, output_dim=1, scope_name=self.scope_name ,
                                                layer1_size=80, layer2_size=50)
        return x, y_pred



class TFLinear(TFRoiBasedModel):

    def __init__(self, tasks, feature_extractor=None):
        super(TFLinear, self).__init__(tasks, feature_extractor)

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'learned_by_roi_70s',
                                        'linear_weights_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))

            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES, output_dim=1, scope_name='lin_reg')
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lin_reg')]



class TFLinearAveraged(TFRoiBasedModel):
    def __init__(self, tasks, feature_extractor=None):
        super(TFLinearAveraged, self).__init__(tasks, feature_extractor)

    def _load(self):
        super(TFLinearAveraged, self)._load()
        return True

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'averaged_weights_70s',
                                        'linear_weights_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))

            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES, output_dim=1, scope_name='lin_reg_avg')
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lin_reg_avg')]


class TFLinearFSF(TFRoiBasedModel):

    def __init__(self, tasks, feature_extractor=None):
        super(TFLinearFSF, self).__init__(tasks, feature_extractor)

    def _load(self):
        super(TFLinearFSF, self)._load()
        self.global_features = demean_and_normalize(self.spatial_filters_raw)
        return True

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'learned_by_roi_filters_as_features70s',
                                        'linear_weights_fsf_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))

            self._weights[task] = weights

    def get_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES+ NUM_SPATIAL_FILTERS, output_dim=1, scope_name='lin_reg_fsf')
        return x, y_pred

    def get_trainable_variables(self):
        return [v for v in tf.trainable_variables() if v in
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lin_reg_fsf')]


    def preprocess_features(self, subject_features):
        subject_features = np.transpose(subject_features)
        subject_features = np.concatenate((subject_features, self.spatial_filters_raw), axis=1)
        subject_features = demean_and_normalize(subject_features)

        return subject_features


class FeatureExtractor:
    """
    This class is used for performing feature extraction from rfmri input files (time-series).
    The feature extractor input is a 2D matrix containing TimePoints values for each brain vertex,
    and outputs a 2D matrix containing the values for 108 features for each cortical vertex.   
    """

    def __init__(self):
        self.matrices = dict()
        self.is_loaded = False
        self.bm = get_bm('cortex')

    def load(self):
        arr = np.load(definitions.SC_CLUSTERS_PATH)
        self.matrices['SC_CLUSTERS'] = arr
        arr = np.load(definitions.ICA_LR_MATCHED_PINV_PATH)
        self.matrices['PINV_ICA_LR_MATCHED'] = arr
        self.is_loaded = True
        return

    def get_features(self,subject):
        """
        returns the features matrix for subject by extracting the features fron the subject's rfmri input,
        or loading it from a saved features file.
        """

        if subject.features_exist:
            arr, (series, bm) = open_features_file(subject.features_path)

            return arr, self.bm
        else:
            if not self.is_loaded:
                self.load()
            return self.extract_features(subject)

    def extract_features(self,subject):
        start_extraction_time = time.time()
        rfmri_data, (series, bm) = open_rfmri_input_data(subject.input_path)   # arr is time * 91k
        n_vertices, t = rfmri_data.shape

        # preprocess
        rfmri_data_normalized = variance_normalise(rfmri_data)  # noise variance normalisation
        data = detrend(rfmri_data_normalized)   # subtract linear trend from time series for each vertex

        # dual regression: get individual cortical spatial maps:
        # step 1 - get subject's individual time series for each by network
        ts_by_network = np.dot(data, self.matrices['PINV_ICA_LR_MATCHED'])  # time x 76
        # step 2 - get subject's individual cortical spatial maps
        LHRH = fsl_glm(ts_by_network, data).transpose() # 91k x 76
        # create spatial maps for the whole brain
        ROIS = np.zeros([STANDARD_BM.N_TOTAL_VERTICES, 108])
        #  left hemisphere networks
        ROIS[: STANDARD_BM.N_LH, : 38] = LHRH[: STANDARD_BM.N_LH, : 38]
        # right hemisphere networks
        ROIS[STANDARD_BM.N_LH : STANDARD_BM.N_CORTEX , 38 :76] = \
            LHRH[STANDARD_BM.N_LH : STANDARD_BM.N_CORTEX , 38 :76]
        # for subcortical networks - use data of group
        ROIS[:, 76:] = self.matrices['SC_CLUSTERS']
        rfmri_data_normalized = demean(rfmri_data_normalized)  # remove mean from each column #TODO is it needed?
        # multiple regression - characteristic time series for each network
        T2 = np.dot(np.linalg.pinv(ROIS), rfmri_data_normalized.transpose()) # 108 x time
        # get the features - correlation coefficient for each vertex with each netwrok
        features_map = np.dot(normalize(T2, axis=1), normalize(rfmri_data_normalized, axis=0))
        features_map = features_map[:, : STANDARD_BM.N_CORTEX]
        save_to_dtseries(subject.features_path, self.bm, features_map)

        end_extraction_time = time.time()

        print("features extracted for subject {0} in {1:.1f} seconds".format(subject.subject_id, end_extraction_time-start_extraction_time))

        return features_map, self.bm


class MemFeatureExtractor(FeatureExtractor):

    def __init__(self, features_mat, subjects, map={}):
        super(MemFeatureExtractor, self).__init__()
        self.matrices['all features'] = features_mat
        self.subjects_mapping = map if map else {subject : int(subject.subject_id)-1 for subject in subjects}

    def get_features(self,subject):
        mat = self.matrices['all features']
        features = np.squeeze(get_subjects_features_from_matrix(mat, [subject], mapping=self.subjects_mapping))
        if np.size(features, axis=1) == NUM_FEATURES:
            features = np.transpose(features)
        if np.size(features, axis=0) != NUM_FEATURES:
            raise PitecaError("features file must include {} features".format(NUM_FEATURES))
        return features, self.bm




available_models = {
    'Linear by ROI averaged betas': TFLinearAveraged,
    'Linear by ROI': TFLinear,
    'Linear by ROI with group connectivity features': TFLinearFSF,
    'MLP by ROI' : NN2lhModel,
    'MLP by ROI with group connectivity features' : NN2lhModelWithFiltersAsFeatures,}

available_models_keys = list(available_models.keys())
available_models_keys.sort()


def model_factory(model, tasks, fe=None):
    if isinstance(model, str):
        if model not in available_models:
            raise ValueError("There is no prediction model named {}".format(model))
        my_model_class = available_models[model]
    elif issubclass(model, IModel):
        my_model_class = model
    else:
        raise ValueError("Argument is not an IModel")
    my_model = my_model_class(tasks, fe)
    return my_model







