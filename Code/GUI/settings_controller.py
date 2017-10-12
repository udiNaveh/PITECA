"""
This module handles the configurable settings of PITECA that can be confidured by user in the settings tab.
It provides getter functions for other module to know the current settings.
"""

import configparser
import os
import definitions
from sharedutils import dialog_utils, constants



features_folder = None
prediction_outputs_folder = None
analysis_results_folder = None
model = None


def get_features_folder():
    return features_folder


def get_prediction_outputs_folder():
    return prediction_outputs_folder


def get_analysis_results_folder():
    return analysis_results_folder


def get_model():
    return model


class SettingsController:
    """
    Controls the settings.ini file, that defines PITECA's setting.
    Changes this file according to ui events in the settings tab.
    """

    def __init__(self, ui):
        self.ui = ui

        global features_folder
        global prediction_outputs_folder
        global analysis_results_folder
        global model

        # init current settings
        self.config = configparser.ConfigParser()
        if os.path.exists(definitions.SETTINGS_PATH):
            self.config.read(definitions.SETTINGS_PATH)
        if constants.SETTINGS_TITLE not in self.config:
            self.config[constants.SETTINGS_TITLE] = {}

        default_values = {constants.SETTINGS_FEATURES_FOLDER: definitions.DEFAULT_EXTRACTED_FEATURES_DIR,
                          constants.SETTINGS_PREDICTIONS_OUTPUT: definitions.DEFAULT_PREDICTIONS_DIR,
                          constants.SETTINGS_ANALYSIS_OUTPUT: definitions.DEFAULT_ANLYSIS_DIR,
                          constants.SETTINGS_MODEL: definitions.DEFAULT_MODEL
                          }

        for key, value in default_values.items():

            if key not in self.config[constants.SETTINGS_TITLE] or \
                    not self.config[constants.SETTINGS_TITLE][key]:
                self.config.set(constants.SETTINGS_TITLE, key, value)

        features_folder = self.config[constants.SETTINGS_TITLE][constants.SETTINGS_FEATURES_FOLDER]
        prediction_outputs_folder = self.config[constants.SETTINGS_TITLE][constants.SETTINGS_PREDICTIONS_OUTPUT]
        analysis_results_folder = self.config[constants.SETTINGS_TITLE][constants.SETTINGS_ANALYSIS_OUTPUT]
        model = self.config[constants.SETTINGS_TITLE][constants.SETTINGS_MODEL]

        self.ui.featuresFolderLineEdit.setText(features_folder)
        self.ui.predictionOutputFolderLineEdit.setText(prediction_outputs_folder)
        self.ui.analysisOutputFolderLineEdit.setText(analysis_results_folder)
        self.ui.ModelComboBox.setCurrentText(model)

        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)


    def set_features_folder(self):
        """
        The function to be called when user changes the features folder on the settings tab.
        """

        global features_folder
        dir = features_folder if features_folder is not None else definitions.ROOT_DIR
        features_folder = dialog_utils.browse_dir(self.ui.featuresFolderLineEdit, dir)
        self.config.set(constants.SETTINGS_TITLE, constants.SETTINGS_FEATURES_FOLDER, features_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_output_predictions_folder(self):
        """
        The function to be called when user changes the output predictions folder on the settings tab.
        """

        global prediction_outputs_folder
        dir = prediction_outputs_folder if prediction_outputs_folder is not None else definitions.ROOT_DIR
        prediction_outputs_folder = dialog_utils.browse_dir(self.ui.predictionOutputFolderLineEdit, dir)
        self.config.set(constants.SETTINGS_TITLE, constants.SETTINGS_PREDICTIONS_OUTPUT, prediction_outputs_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_output_analysis_folder(self):
        """
        The function to be called when user changes the analysis on the settings tab.
        """

        global analysis_results_folder
        dir = analysis_results_folder if analysis_results_folder is not None else definitions.ROOT_DIR
        analysis_results_folder = dialog_utils.browse_dir(self.ui.analysisOutputFolderLineEdit, dir)
        self.config.set(constants.SETTINGS_TITLE, constants.SETTINGS_ANALYSIS_OUTPUT, analysis_results_folder)
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

    def set_model(self):
        """
        The function to be called when user changes the model in settings tab.
        """

        global model
        key = self.ui.ModelComboBox.currentText()
        self.config.set(constants.SETTINGS_TITLE, constants.SETTINGS_MODEL, key)
        model = key
        with open(definitions.SETTINGS_PATH, 'w') as configfile:
            self.config.write(configfile)

