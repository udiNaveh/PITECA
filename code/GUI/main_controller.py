from GUI.predict_controller import PredictController
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QFile, QIODevice
from GUI.views import Ui_MainView
'''
Connect buttons from the ui to their functionality
'''


def setup_functionality(ui):
    # For Predict tab
    predictController = PredictController(ui)
    ui.browseInputFilesButton.clicked.connect(lambda: predictController.onBrowseInputFilesClicked())
    ui.browseOutputDirButton.clicked.connect(lambda: predictController.onBrowseOutputDirClicked())
    ui.runPredictButton.clicked.connect(lambda: predictController.onRunPredictClicked())


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