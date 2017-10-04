from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from GUI.popups.predict_working_dlg_view import Ui_PredictWorkingDlg
from GUI.predict_working_thread import PredictWorkingThread
from sharedutils import dialog_utils, constants


class ResultsDlg(QtWidgets.QDialog):

    def __init__(self, ui, label_text, folder):
        super(ResultsDlg, self).__init__()
        ui.msgLabel.setText(label_text)
        ui.openInFolderButton.clicked.connect(lambda: print('hello'))
        ui.okButton.clicked.connect(lambda: print('hello hello'))


