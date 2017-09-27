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

    default_values = { 'FeaturesFolder' : definitions.DEFAULT_EXTRACTED_FEATURES_DIR,
      'PredictionOutputsFolder' : definitions.DEFAULT_PREDICTIONS_DIR,
      'AnalysisOutputFolder' :  definitions.DEFAULT_ANLYSIS_DIR
    }

    def __init__(self, ui):
        self.ui = ui

        global features_folder
        global prediction_outputs_folder
        global analysis_results_folder

        # init current settings
        self.config = configparser.ConfigParser()
        self.config.read(definitions.SETTINGS_PATH)


        for key, value in SettingsController.default_values.items():
            if key not in self.config['SETTINGS'] or \
                    not self.config['SETTINGS'][key]:
                self.config.set('SETTINGS', key, value)

        features_folder = self.config['SETTINGS']['FeaturesFolder']
        prediction_outputs_folder = self.config['SETTINGS']['PredictionOutputsFolder']
        analysis_results_folder = self.config['SETTINGS']['AnalysisOutputFolder']

        self.ui.featuresFolderLineEdit.setText(features_folder)
        self.ui.predictionOutputFolderLineEdit.setText(prediction_outputs_folder)
        self.ui.analysisOutputFolderLineEdit.setText(analysis_results_folder)

        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)


    def set_features_folder(self):
        global features_folder
        dir = features_folder if features_folder is not None else definitions.ROOT_DIR
        features_folder = dialog_utils.browse_dir(self.ui.featuresFolderLineEdit, dir)
        self.config.set('SETTINGS', 'FeaturesFolder', features_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_output_predictions_folder(self):
        global prediction_outputs_folder
        dir = prediction_outputs_folder if prediction_outputs_folder is not None else definitions.ROOT_DIR
        prediction_outputs_folder = dialog_utils.browse_dir(self.ui.predictionOutputFolderLineEdit, dir)
        self.config.set('SETTINGS', 'PredictionOutputsFolder', prediction_outputs_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_output_analysis_folder(self):
        global analysis_results_folder
        dir = analysis_results_folder if analysis_results_folder is not None else definitions.ROOT_DIR
        analysis_results_folder = dialog_utils.browse_dir(self.ui.analysisOutputFolderLineEdit, dir)
        self.config.set('SETTINGS', 'AnalysisOutputFolder', analysis_results_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    # TODO: add a button to reset to default settings
