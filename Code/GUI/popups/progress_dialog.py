from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from GUI.popups.progressBarView import Ui_ProgressBar
from GUI.progress_thread import ProgressThread

class ProgressDlg(QtWidgets.QDialog):

    def __init__(self, model, subjects):
        super(ProgressDlg, self).__init__()
        self.prediction_model = model
        self.subjects = subjects

    def create_ui(self):
        progress_bar_ui = Ui_ProgressBar()
        progress_bar_ui.setupUi(self)
        self.progress_thread = ProgressThread(self.prediction_model, self.subjects, progress_bar_ui.progressBar)
        progress_bar_ui.pushButton.clicked.connect(lambda: self.progress_thread.terminate())

    def closeEvent(self, event):
        self.progress_thread.terminate()
        event.accept

    def start_progress(self):
        self.progress_thread.start()