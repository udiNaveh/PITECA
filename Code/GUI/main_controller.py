from GUI.predict_controller import PredictController
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QFile, QIODevice
from GUI.views import Ui_MainView
from GUI.analyze_controller import AnalyzeController
from sharedutils import constants

'''
The main of PITECA project.
Runs the app and connects user events on the Main Window of PITECA to their functionality.
'''

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


if __name__ == "__main__":
    pass
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainView = QtWidgets.QMainWindow()
    ui = Ui_MainView()
    ui.setupUi(MainView)
    setup_functionality(ui)
    MainView.show()
    sys.exit(app.exec_())