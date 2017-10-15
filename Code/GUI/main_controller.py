import sys
import threading
import traceback

from GUI.predict_controller import PredictController
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread

import GUI.globals as gb
from GUI.views import Ui_MainView
from GUI.analyze_controller import AnalyzeController
from GUI.settings_controller import SettingsController
from sharedutils import constants, dialog_utils
import definitions


'''
The main of PITECA project.
Runs the app and connects user events on the Main Window of PITECA to their functionality.
'''

def setup_thread_excepthook():
    """
    Workaround for `sys.excepthook` thread bug from:
    http://bugs.python.org/issue1230540

    Call once from the main thread before creating any threads.
    """

    init_original = threading.Thread.__init__

    def init(self, *args, **kwargs):

        init_original(self, *args, **kwargs)
        run_original = self.run

        def run_with_except_hook(*args2, **kwargs2):
            try:
                run_original(*args2, **kwargs2)
            except Exception:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init


def setup_functionality(ui):
    # For Predict tab
    predictController = PredictController(ui)
    ui.browseInputFilesButton.clicked.connect(lambda: predictController.onBrowseInputFilesClicked())
    ui.runPredictButton.clicked.connect(lambda: predictController.onRunPredictClicked())

    # For Analysis tab
    analyzeController = AnalyzeController(ui)
    ui.domainComboBox.currentIndexChanged.connect(lambda: analyzeController.update_tasks())
    ui.browsePredictedButton.clicked.connect(lambda: analyzeController.onPredictedInputBrowseButtonClicked())
    ui.browseActualButton.clicked.connect(lambda: analyzeController.onActualInputBrowseButtonClicked())
    ui.runAnalyzeButton.clicked.connect(lambda: analyzeController.onRunAnalysisButtonClicked())
    ui.runCompareButton.clicked.connect(lambda: analyzeController.onRunComparisonButtonClicked())

    # For Settings tab
    settingsController = SettingsController(ui)
    ui.featuresFolderButton.clicked.connect(lambda: settingsController.set_features_folder())
    ui.predictionOutputFolderButton.clicked.connect(lambda: settingsController.set_output_predictions_folder())
    ui.analysisOutputFolderButton.clicked.connect(lambda: settingsController.set_output_analysis_folder())
    ui.ModelComboBox.currentIndexChanged.connect(lambda: settingsController.set_model())

def piteca_excepthook(exctype, value, tb):
    """
    A method to catch all unhandled exception during PITECA's run .
    :param exctype: the type of exception
    :param value: the message of the exception (use str(value))
    :param tb: traceback
    """

    # if not gb.should_exit_on_error:
    # # If we are on main thread but don't want to close PITECA
    #     dialog_utils.print_error(constants.UNEXPECTED_EXCEPTION_MSG[:-1] + ": " + str(value))
    #     return

    # Show user the full error value only if it is a PITECA error
    if exctype == definitions.PitecaError:
        msg = str(value.message)
    else:
        msg = constants.UNEXPECTED_EXCEPTION_MSG

    if int(QThread.currentThreadId()) == main_thread_id:
        definitions.print_in_debug(value)
        traceback.print_tb(tb)
        dialog_utils.print_error(msg + ". PITECA will be now closed")
        sys.exit()
    else:
        # The exception_occurred_sig should be defined in every thread class in PITECA
        definitions.print_in_debug(value)
        definitions.print_in_debug(exctype)
        traceback.print_tb(tb)
        QThread.currentThread().exception_occurred_sig.emit(msg)


if __name__ == "__main__":
    pass
    main_thread_id = int(QThread.currentThreadId())
    setup_thread_excepthook()
    sys.excepthook = piteca_excepthook
    app = QtWidgets.QApplication(sys.argv)
    MainView = QtWidgets.QMainWindow()
    ui = Ui_MainView()
    ui.setupUi(MainView)
    setup_functionality(ui)
    MainView.show()
    sys.exit(app.exec_())