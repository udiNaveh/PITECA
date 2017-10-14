import pickle
import numpy as np
from sklearn.preprocessing import normalize
import time
from sharedutils.io_utils import *
from sharedutils.linalg_utils import *
from sharedutils.subject import *
import definitions
from model.data_manager import get_subjects_features_from_matrix


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



class MemFeatureExtractor(FeatureExtractor):
    """
    An implementation of Feature Extractor that returns extracted features from a matrix.
    This is useful in training, to avoid loading the features from the cifti files, which takes time.  
    """

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
            raise definitions.PitecaError("features file must include {} features".format(NUM_FEATURES))
        return features, self.bm