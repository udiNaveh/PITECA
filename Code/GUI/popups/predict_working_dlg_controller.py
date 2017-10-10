from PyQt5 import QtWidgets

from GUI.popups.predict_working_dlg_view import Ui_PredictWorkingDlg
from GUI.predict_working_thread import PredictWorkingThread
from sharedutils import dialog_utils
from GUI import settings_controller
import GUI.globals as gb


class PredictWorkingDlg(QtWidgets.QDialog):
    """
    The dialog shown during prediction progress
    """

    def __init__(self, model, subjects):
        super(PredictWorkingDlg, self).__init__()
        self.prediction_model = model
        self.subjects = subjects
        self.progress_bar_ui = Ui_PredictWorkingDlg()
        self.progress_bar_ui.setupUi(self)
        self.progress_thread = PredictWorkingThread(self.prediction_model, self.subjects, self.progress_bar_ui.progressBar)
        self.progress_thread.finished_sig.connect(lambda: self.on_predict_finish())
        self.closeEvent = lambda event: self.onClose(event)

    def on_predict_finish(self):
        """
        The function to be called when the thread that does the prediction work finishes.
        Closes the progress dialog and opens the results dialog.
        """

        self.close()

        if len(self.subjects) == 1 and len(self.prediction_model.tasks) == 1:
            filepath = gb.curr_cifti_filename
        else:
            filepath = None

        dialog_utils.report_results(
            "Done! Predicted files are saved in {}".format(settings_controller.get_prediction_outputs_folder()),
            settings_controller.get_prediction_outputs_folder(), filepath)

    def onClose(self, event):
        """
        The function to be called when user quits the progress dialog.
        """
        self.progress_thread.terminate()
        self.close()
    # An implementation that asks the user if really should quit:
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