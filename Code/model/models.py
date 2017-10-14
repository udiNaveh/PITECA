"""
This module contains the core logic of the prediction models in PITECA,
which is to predict and generate activation maps for subjects based on their
input rfmri data. Several models are included here (and can be added), all implementing the abstract base class IModel.

"""
import os.path
import inspect
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
     Abstract base class for all models
    """

    def __init__(self, tasks):
        self.tasks = tasks
        super().__init__()

    @abstractmethod
    def predict(self, subject):
        pass


class TFRoiBasedModel(IModel):
    """
    Base abstract class for all roi-based models implemented in tensorflow (currently all available models).
    These models divide the cortex vertices into 49 rois, where each roi has its own predictive model.
    The predicted value for each vertex is determined using the model of the roi to which it belongs.
    Note however that if soft filters are used, a vertex's relation to the different roi's is describes as a 
    (usually sparse) probability vector. Thus a vertex precition is a weighted average
    of the different predictions it gets from the different roi models.
    """

    def __init__(self, tasks, feature_extractor = None):
        super(TFRoiBasedModel, self).__init__(tasks)
        self._weights = {}
        self.feature_extractor = FeatureExtractor() if feature_extractor is None else feature_extractor
        self.is_loaded = False
        self.scope_name = ''

    def _load(self):
        spatial_filters_raw, (series, bm) = cifti.read(definitions.ICA_LOW_DIM_PATH)
        spatial_filters_raw = np.transpose(spatial_filters_raw[:, STANDARD_BM.CORTEX])
        soft_filters = softmax(spatial_filters_raw.astype(float) / TEMPERATURE)
        soft_filters[soft_filters < FILTERS_EPSILON] = 0.0
        soft_filters[:, 2] = 0
        soft_filters /= np.reshape(np.sum(soft_filters, axis=1), [STANDARD_BM.N_CORTEX, 1])
        hard_filters = np.round(softmax(spatial_filters_raw.astype(float) * 1000))
        hard_filters[spatial_filters_raw < SPATIAL_FILTERS_THRESHOLD] = 0
        self.spatial_filters = soft_filters
        self.spatial_filters_hard = hard_filters
        self.x, self.y_pred, self.variables = self.build_model()
        self.load_weights()
        self.spatial_filters_raw = spatial_filters_raw
        self.global_features = None
        return True

    @abstractmethod
    def load_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_input_output_placeholders(self):
        raise NotImplementedError()

    def build_model(self):
        """
        build the TF graph for model and return the o
        :return: x - the input placeholder
                y_pred - the output placeholder
                 variables - a list of all trainale varianle tensors in the computational graph
        """
        x, y_pred = self._get_input_output_placeholders()
        variables = self._get_trainable_variables()
        return x, y_pred, variables


    def _get_trainable_variables(self):
        """        
        :return: The trainable variables' tensors from model's computational graph. 
        """
        return [v for v in tf.trainable_variables() if v in
                      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)]

    def preprocess_features(self, subject_features):
        """
        preprocessing of the extracted features before they are fed as input into the prediction model
        :param subject_features: a 2-dimensional array representing a subject's features across the cortex
        :return: subject_features after preprocessing
        """
        if np.size(subject_features, axis=0) == NUM_FEATURES:
            subject_features = np.transpose(subject_features)
        if np.shape(subject_features) != (STANDARD_BM.N_CORTEX, NUM_FEATURES):
            raise ValueError("Extracted features should be of shape ({0},{1}), but is instead {2}".format(
                STANDARD_BM.N_CORTEX, NUM_FEATURES, subject_features.shape
            ))
        subject_features = demean_and_normalize(subject_features)
        if self.global_features is not None:
            subject_features = np.concatenate([subject_features, self.global_features], axis=1)
        return subject_features

    def get_spatial_filters(self, subject_feats):
        return self.spatial_filters

    def predict(self, subject, save = True):
        """
        Generates the predicted activation maps for subject for all tasks in self.tasks.
        saves each map in a .dtseries.nii file.
        
        :param subject: an object of type Subject 
        :param save: bool - whether or not to save the predicted activation into .dtseries.nii files
        :return: a Tuple(subject_predictions, prediction_paths), with
                subject_predictions : a dictionary mapping tasks to prediction matrices
                prediction_paths : a dictionary mapping tasks to paths of saved predictions in .dtseries.nii files
        """
        if not self.is_loaded:
            self.is_loaded = self._load()
        arr, bm = self.feature_extractor.get_features(subject)
        subject_features = self.preprocess_features(arr)
        spatial_filters = self.get_spatial_filters(subject_features)
        subject_predictions, prediction_paths = {}, {}
        with tf.Session() as sess:
            for task in self.tasks:
                subject_task_prediction = np.zeros([1, STANDARD_BM.N_CORTEX])
                start = time.time()
                for j in range(np.size(spatial_filters, axis=1)):
                    if j in self._weights[task]:
                        roi_indices = spatial_filters[: STANDARD_BM.N_CORTEX, j] > 0
                        roi_weight = spatial_filters[:, j][roi_indices]
                        roi_features = subject_features[roi_indices]
                        roi_weights = self._weights[task][j]
                        roi_feed_dict = union_dicts({self.x: roi_features},
                                                       {tensor: roi_weights[i] for i, tensor in enumerate(self.variables)})
                        roi_prediction = np.squeeze(sess.run(self.y_pred, feed_dict=roi_feed_dict))
                        subject_task_prediction[:,roi_indices] += roi_weight * roi_prediction
                subject_predictions[task] = subject_task_prediction
                end = time.time()
                definitions.print_in_debug("prediction of task {0} subject {1} took {2:.3f}seconds".format(
                    task.full_name, subject.subject_id, end-start))
                if save:
                    prediction_paths[task] = save_to_dtseries(subject.get_predicted_task_filepath(task), bm,
                                                              subject_task_prediction)
        return subject_predictions, prediction_paths


class NN2lhModel(TFRoiBasedModel):
    """
    An MLP model with two hidden layers, that uses subjects' individual functional features
    to predict task activation.
    """

    def __init__(self, tasks, feature_extractor=None, layer1_size=HL1_SIZE, layer2_size = HL2_SIZE, scope_name = 'NN2lh'):
        super(NN2lhModel, self).__init__(tasks, feature_extractor)
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.scope_name = scope_name

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.NN_WEIGHTS_DIR, '2hl_70s',
                           'nn_2hl_no_roi_normalization_70s_weights_{0}_all_filters.pkl'.format(task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def _get_input_output_placeholders(self):
        x, y, y_pred = \
            regression_with_two_hidden_layers_build(input_dim= NUM_FEATURES, output_dim=1, scope_name=self.scope_name,
                                                layer1_size=self.layer1_size, layer2_size=self.layer2_size)
        return x, y_pred



class NN2lhModelWithFiltersAsFeatures(TFRoiBasedModel):
    """
    An MLP model with two hidden layers, that uses subjects' individual functional features
    and combined with global functional features (for each vertex) to predict task activation.
    """

    def __init__(self, tasks, feature_extractor=None, layer1_size= HL1_SIZE, layer2_size = HL2_SIZE, scope_name = 'NN2lhfsf'):
        super(NN2lhModelWithFiltersAsFeatures, self).__init__(tasks, feature_extractor)
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.scope_name = scope_name

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

    def _get_input_output_placeholders(self):
        input_dim = NUM_FEATURES + NUM_SPATIAL_FILTERS
        x, y, y_pred = \
            regression_with_two_hidden_layers_build(input_dim= input_dim, output_dim=1, scope_name=self.scope_name,
                                                layer1_size=self.layer1_size, layer2_size=self.layer2_size)
        return x, y_pred


class TFLinear(TFRoiBasedModel):
    """
    A basic linear model that uses subjects' individual functional features
    for each vertex to predict task activation.
    """

    def __init__(self, tasks, feature_extractor=None, scope_name= 'linear_basic'):
        super(TFLinear, self).__init__(tasks, feature_extractor)
        self.scope_name = scope_name

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'learned_by_roi_70s',
                                        'linear_weights_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def _get_input_output_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES, output_dim=1, scope_name=self.scope_name)
        return x, y_pred



class TFLinearAveraged(TFRoiBasedModel):
    """
    A basic linear model that uses subjects' individual functional features, using weights that were learned by
    averaging weights aquired from linear regression on each subject.
    Uses hard roi filters, thus ~4500 cortical vertices with poor signal do not have prediction, as in the original
    model by Tavor et al. 2016
    """

    def __init__(self, tasks, feature_extractor=None, scope_name= 'linear_avg'):
        super(TFLinearAveraged, self).__init__(tasks, feature_extractor)
        self.scope_name = scope_name

    def _load(self):
        super(TFLinearAveraged, self)._load()
        self.spatial_filters = self.spatial_filters_hard # TODO
        return True

    def load_weights(self):
        for task in self.tasks:
            weights_path = os.path.join(definitions.LINEAR_WEIGHTS_DIR, 'averaged_weights_70s',
                                        'linear_weights_{}.pkl'.format(
                                            task.full_name))
            weights = pickle.load(open(weights_path, 'rb'))
            self._weights[task] = weights

    def _get_input_output_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES, output_dim=1, scope_name=self.scope_name)
        return x, y_pred


class TFLinearFSF(TFRoiBasedModel):
    """
    A linear model that uses subjects' individual functional features
    combined with global functional features (for each vertex) to predict task activation.
    """
    def __init__(self, tasks, feature_extractor=None, scope_name = 'linear_fsf'):
        super(TFLinearFSF, self).__init__(tasks, feature_extractor)
        self.scope_name = scope_name

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

    def _get_input_output_placeholders(self):
        x, y, y_pred = \
            linear_regression_build(input_dim=NUM_FEATURES+ NUM_SPATIAL_FILTERS, output_dim=1, scope_name=self.scope_name)
        return x, y_pred


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
        returns the features matrix for subject by extracting the features from the subject's rfmri input,
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
        rfmri_data, (series, bm) = open_rfmri_input_data(subject.input_path)

        # preprocess
        rfmri_data_normalized = variance_normalise(rfmri_data)  # noise variance normalisation
        data = detrend(rfmri_data_normalized)   # subtract linear trend from time series for each vertex

        # dual regression: get individual cortical spatial maps:
        # step 1 - get subject's individual time series for each by network
        ts_by_network = np.dot(data, self.matrices['PINV_ICA_LR_MATCHED'])  # time x 76
        # step 2 - get subject's individual cortical spatial maps
        LHRH = fsl_glm(ts_by_network, data).transpose()
        # create spatial maps for the whole brain
        ROIS = np.zeros([STANDARD_BM.N_TOTAL_VERTICES, 108])
        #  left hemisphere networks
        ROIS[: STANDARD_BM.N_LH, : 38] = LHRH[: STANDARD_BM.N_LH, : 38]
        # right hemisphere networks
        ROIS[STANDARD_BM.N_LH : STANDARD_BM.N_CORTEX , 38 :76] = \
            LHRH[STANDARD_BM.N_LH : STANDARD_BM.N_CORTEX , 38 :76]
        # for subcortical networks - use data of group
        ROIS[:, 76:] = self.matrices['SC_CLUSTERS']
        rfmri_data_normalized = demean(rfmri_data_normalized)  # remove mean from each column
        # multiple regression - characteristic time series for each network
        T2 = np.dot(np.linalg.pinv(ROIS), rfmri_data_normalized.transpose()) # 108 x time
        # get the features - correlation coefficient for each vertex with each netwrok
        features_map = np.dot(normalize(T2, axis=1), normalize(rfmri_data_normalized, axis=0))

        # save and return
        features_map = (features_map[:, : STANDARD_BM.N_CORTEX]).astype(np.float32)
        save_to_dtseries(subject.features_path, self.bm, features_map)
        end_extraction_time = time.time()
        definitions.print_in_debug("features extracted for subject {0} in {1:.1f} seconds".format(
            subject.subject_id, end_extraction_time-start_extraction_time))

        return features_map, self.bm





available_models = {
    'Linear by ROI averaged betas': TFLinearAveraged,
    'Linear by ROI': TFLinear,
    'Linear by ROI with group functional features': TFLinearFSF,
    'MLP by ROI' : NN2lhModel,
    'MLP by ROI with group functional features' : NN2lhModelWithFiltersAsFeatures,}

available_models_keys = list(available_models.keys())
available_models_keys.sort()


def model_factory(model, tasks, **kwars):
    if isinstance(model, str):
        if model not in available_models:
            raise ValueError("There is no prediction model named {}".format(model))
        my_model_class = available_models[model]
    elif inspect.isclass(model) and issubclass(model, IModel):
        my_model_class = model
    else:
        raise ValueError("Argument is not an IModel")
    my_model = my_model_class(tasks, **kwars)
    return my_model







