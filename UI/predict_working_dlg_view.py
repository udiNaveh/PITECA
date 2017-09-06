# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Sadna\src\git\PITECA\UI\progress_bar.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PredictWorkingDlg(object):
    def setupUi(self, PredictWorkingDlg):
        PredictWorkingDlg.setObjectName("PredictWorkingDlg")
        PredictWorkingDlg.resize(234, 64)
        self.verticalLayout = QtWidgets.QVBoxLayout(PredictWorkingDlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.progressBar = QtWidgets.QProgressBar(PredictWorkingDlg)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.label = QtWidgets.QLabel(PredictWorkingDlg)
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)

        self.retranslateUi(PredictWorkingDlg)
        QtCore.QMetaObject.connectSlotsByName(PredictWorkingDlg)

    def retranslateUi(self, PredictWorkingDlg):
        _translate = QtCore.QCoreApplication.translate
        PredictWorkingDlg.setWindowTitle(_translate("PredictWorkingDlg", "Dialog"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PredictWorkingDlg = QtWidgets.QDialog()
    ui = Ui_PredictWorkingDlg()
    ui.setupUi(PredictWorkingDlg)
    PredictWorkingDlg.show()
    sys.exit(app.exec_())

