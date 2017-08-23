from PyQt5 import QtWidgets
from GUI.popups import analysis_working_dlg_view


class AnalysisWorkingDlg(QtWidgets.QDialog):

    def __init__(self):
        super(AnalysisWorkingDlg, self).__init__()
        ui = analysis_working_dlg_view.Ui_AnalysisWorkingDlg()
        ui.setupUi(self)