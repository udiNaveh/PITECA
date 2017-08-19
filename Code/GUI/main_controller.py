from GUI.predict_controller import PredictController
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QFile, QIODevice
from GUI.views import Ui_MainView
from GUI.analyze_controller import AnalyzeController
from sharedutils import constants, dialog_utils
import sys
import threading


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
    ui.browseOutputDirButton.clicked.connect(lambda: predictController.onBrowseOutputDirClicked())
    ui.runPredictButton.clicked.connect(lambda: predictController.onRunPredictClicked())

    # For Analysis tab
    analyzeController = AnalyzeController(ui)
    ui.domainComboBox.currentIndexChanged.connect(lambda: analyzeController.update_tasks())
    ui.browsePredictedButton.clicked.connect(lambda: analyzeController.onPredictedInputBrowseButtonClicked())
    ui.browseActualButton.clicked.connect(lambda: analyzeController.onActualInputBrowseButtonClicked())
    ui.runAnalyzeButton.clicked.connect(lambda: analyzeController.onRunAnalysisButtonClicked())
    ui.runCompareButton.clicked.connect(lambda: analyzeController.onRunComparisonButtonClicked())


def piteca_excepthook(exctype, value, tb):
    dialog_utils.print_error(str(value) + ". PITECA will be now closed")
    sys.exit()


if __name__ == "__main__":
    pass
    setup_thread_excepthook()
    sys.excepthook = piteca_excepthook
    app = QtWidgets.QApplication(sys.argv)
    MainView = QtWidgets.QMainWindow()
    ui = Ui_MainView()
    ui.setupUi(MainView)
    setup_functionality(ui)
    MainView.show()
    sys.exit(app.exec_())