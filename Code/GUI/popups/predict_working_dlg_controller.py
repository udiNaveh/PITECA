from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from GUI.popups.predict_working_dlg_view import Ui_PredictWorkingDlg
from GUI.predict_working_thread import PredictWorkingThread


class PredictWorkingDlg(QtWidgets.QDialog):

    def __init__(self, model, subjects):
        super(PredictWorkingDlg, self).__init__()
        self.prediction_model = model
        self.subjects = subjects
        progress_bar_ui = Ui_PredictWorkingDlg()
        progress_bar_ui.setupUi(self)
        self.progress_thread = PredictWorkingThread(self.prediction_model, self.subjects, progress_bar_ui.progressBar)
        progress_bar_ui.pushButton.clicked.connect(lambda: self.progress_thread.terminate())

    def closeEvent(self, event):
        self.progress_thread.terminate()
        event.accept

    def start_progress(self):
        self.progress_thread.start()