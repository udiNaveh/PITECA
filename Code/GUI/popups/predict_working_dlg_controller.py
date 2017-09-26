from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from GUI.popups.predict_working_dlg_view import Ui_PredictWorkingDlg
from GUI.predict_working_thread import PredictWorkingThread
from sharedutils import dialog_utils, constants


class PredictWorkingDlg(QtWidgets.QDialog):

    def __init__(self, model, subjects):
        super(PredictWorkingDlg, self).__init__()
        self.prediction_model = model
        self.subjects = subjects
        self.progress_bar_ui = Ui_PredictWorkingDlg()
        self.progress_bar_ui.setupUi(self)
        self.progress_thread = PredictWorkingThread(self.prediction_model, self.subjects, self.progress_bar_ui.progressBar)
        self.progress_thread.finished_sig.connect(lambda: self.onFinish())

    def onFinish(self):
        self.progress_bar_ui.label.setText("Done!")

    # def closeEvent(self, event):
    #     if self.progress_thread.isFinished():
    #         event.accept()
    #     else:
    #         should_stop = dialog_utils.ask_user(False, constants.QUESTION_TITLE, constants.ARE_YOU_SURE_MSG)
    #         if should_stop:
    #             self.progress_thread.quit()
    #             event.accept()
    #         else:
    #             event.ignore()

    def start_progress(self):
        self.progress_thread.start()