from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from GUI.popups.predict_working_dlg_view import Ui_PredictWorkingDlg
from GUI.predict_working_thread import PredictWorkingThread
from sharedutils import dialog_utils, constants, cmd_utils
import platform, os, subprocess


class ResultsDlg(QtWidgets.QDialog):

    def __init__(self, label_text, folder, exit_on_open_folder, filepath):
        super(ResultsDlg, self).__init__()
        self.label_text = label_text
        self.folder = folder
        self.exit_on_open_folder = exit_on_open_folder
        self.filepath = filepath

    def update_ui(self, ui):
        if self.filepath:
            ui.viewInWbButton.setEnabled(True)
        ui.msgLabel.setText(self.label_text)
        ui.openInFolderButton.clicked.connect(lambda: self.open_in_folder())
        ui.okButton.clicked.connect(lambda: self.close())
        ui.viewInWbButton.clicked.connect(lambda: self.open_in_wb())

    def open_in_folder(self):
        # if self.exit_on_open_folder:
        #     self.close()
        if platform.system() == "Windows":
            os.startfile(self.folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", self.folder])
        else:
            subprocess.Popen(["xdg-open", self.folder])

    def open_in_wb(self):
        cmd_utils.show_maps_in_wb_view(self.filepath)

