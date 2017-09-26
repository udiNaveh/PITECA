import configparser
import definitions
import os
from sharedutils import dialog_utils

features_folder = None
prediction_outputs_folder = None
analysis_results_folder = None

def get_features_folder():
    return features_folder

def get_prediction_outputs_folder():
    return prediction_outputs_folder

def get_analysis_results_folder():
    return analysis_results_folder

class SettingsController:

    def __init__(self, ui):
        self.ui = ui

        global features_folder
        global prediction_outputs_folder
        global analysis_results_folder

        # init current settings
        self.config = configparser.ConfigParser()
        self.config.read(definitions.SETTINGS_PATH)

        features = self.config['SETTINGS']['FeaturesFolder']
        predictions = self.config['SETTINGS']['PredictionOutputsFolder']
        analysis = self.config['SETTINGS']['AnalysisOutputFolder']

        # if settings have never changed before, use PITECA's default (relative paths)
        # else use absolute paths from user
        if self.config['SETTINGS']['isDefaultFeatures'] == 'True':
            features_folder = os.path.join(definitions.DATA_DIR, features)
        else:
            features_folder = features
        if self.config['SETTINGS']['isDefaultPredictions'] == 'True':
            prediction_outputs_folder = os.path.join(definitions.DATA_DIR, predictions)
        else:
            prediction_outputs_folder = predictions
        if self.config['SETTINGS']['isDefaultAnalysis'] == 'True':
            analysis_results_folder = os.path.join(definitions.DATA_DIR, analysis)
        else:
            analysis_results_folder = analysis

        self.ui.featuresFolderLineEdit.setText(features_folder)
        self.ui.predictionOutputFolderLineEdit.setText(prediction_outputs_folder)
        self.ui.analysisOutputFolderLineEdit.setText(analysis_results_folder)


    def set_features_folder(self):
        self.config.set('SETTINGS', 'isDefaultFeatures', 'False')
        global features_folder
        features_folder = dialog_utils.browse_dir(self.ui.featuresFolderLineEdit)
        self.config.set('SETTINGS', 'FeaturesFolder', features_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_output_predictions_folder(self):
        self.config.set('SETTINGS', 'isDefaultPredictions', 'False')
        global prediction_outputs_folder
        prediction_outputs_folder = dialog_utils.browse_dir(self.ui.predictionOutputFolderLineEdit)
        self.config.set('SETTINGS', 'PredictionOutputsFolder', prediction_outputs_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_output_analysis_folder(self):
        self.config.set('SETTINGS', 'isDefaultAnalysis', 'False')
        global analysis_results_folder
        analysis_results_folder = dialog_utils.browse_dir(self.ui.analysisOutputFolderLineEdit)
        self.config.set('SETTINGS', 'AnalysisOutputFolder', analysis_results_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    # TODO: add a button to reset to default settings
