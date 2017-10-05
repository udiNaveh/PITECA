import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.split(CODE_DIR)[0]
CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.ini')

DATA_DIR = os.path.join(ROOT_DIR, 'Data')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'Predictions')
MODELS_DIR = os.path.join(DATA_DIR, 'Models')
LINEAR_MODEL_DIR = os.path.join(MODELS_DIR, 'LinearModel')
EXTRACTED_FEATURES_DIR = os.path.join(DATA_DIR, 'ExtractedFeatures')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'Analysis')
LINEAR_MODEL_BETAS_PATH = os.path.join(LINEAR_MODEL_DIR, 'average_betas_100_subjects_7_tasks.npy')
SC_CLUSTERS_PATH = os.path.join(MODELS_DIR, 'FeatureExtractor', 'SC_clusters.npy')
BM_CORTEX_PATH = os.path.join(MODELS_DIR, 'FeatureExtractor', 'cortex_bm.pkl')
BM_FULL_PATH = os.path.join(MODELS_DIR, 'FeatureExtractor', 'full_bm.pkl')
ICA_LR_MATCHED_PATH = os.path.join(MODELS_DIR, 'FeatureExtractor', 'ica_LR_MATCHED.dtseries.nii')
ICA_LR_MATCHED_PINV_PATH = os.path.join(MODELS_DIR,'FeatureExtractor', 'pinvg.npy')
ICA_LOW_DIM_PATH = os.path.join(MODELS_DIR,'ica_both_lowdim.dtseries.nii')
UI_DIR = os.path.join(ROOT_DIR, 'UI')
PITECA_ICON_PATH = os.path.join(UI_DIR, 'piteca_icon.gif')
CANONICAL_CIFTI_PATH = os.path.join(DATA_DIR, 'canonical.dtseries.nii') # TODO: get the file from Ido and update name here
LOCAL_DATA_DIR = r'D:\Projects\PITECA\Data' #TODO
NN_WEIGHTS_DIR = os.path.join(DATA_DIR, 'Models','NN_Models')

LINEAR_WEIGHTS_DIR = os.path.join(DATA_DIR, 'Models','LinearModels')


#EXTRACTED_FEATURES_DIR = r'D:\Projects\PITECA\Data_for_testing\Features'
SETTINGS_PATH = os.path.join(CODE_DIR, 'settings.ini')

DEFAULT_LEFT_SURFACE_PATH = os.path.join(DATA_DIR, 'Surfaces', r'Q1-Q6_R440.L.midthickness.32k_fs_LR.surf.gii')
DEFAULT_RIGHT_SURFACE_PATH = os.path.join(DATA_DIR, 'Surfaces', r'Q1-Q6_R440.R.midthickness.32k_fs_LR.surf.gii')

DEFAULT_EXTRACTED_FEATURES_DIR = os.path.join(DATA_DIR, 'ExtractedFeatures')
DEFAULT_ANLYSIS_DIR = os.path.join(DATA_DIR, 'Analysis')
DEFAULT_PREDICTIONS_DIR = os.path.join(DATA_DIR, 'Predictions')