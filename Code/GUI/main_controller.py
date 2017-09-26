from GUI.predict_controller import PredictController
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QFile, QIODevice, QThread
from GUI.views import Ui_MainView
from GUI.analyze_controller import AnalyzeController
from GUI.settings_controller import SettingsController
from sharedutils import constants, dialog_utils
import sys
from threading import current_thread
import threading
import GUI.globals as gb
from PyQt5.QtWidgets import QStyleFactory


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

def piteca_excepthook(exctype, value, tb):
    if not gb.should_exit_on_error:
    # If we are on main thread but don't want to close PITECA
        dialog_utils.print_error(constants.UNEXPECTED_EXCEPTION_MSG[:-1] + ": " + str(value))
        print(value)  # TODO: remove this! Here only for development needs
        return
    if int(QThread.currentThreadId()) == main_thread_id:
        dialog_utils.print_error(str(value) + ". PITECA will be now closed")
        sys.exit()
    else:
        # The exception_occurred_sig should be defined in every thread class in PITECA
        print("Inside piteca_excepthook")
        print(value) # TODO: remove this! Here only for development needs
        QThread.currentThread().exception_occurred_sig.emit()


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