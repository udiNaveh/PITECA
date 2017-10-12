# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Sadna\src\git\PITECA\UI\progress_bar.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import definitions
from sharedutils import constants

class Ui_PredictWorkingDlg(object):
    """
    Defines the UI view of the dialog shown during prediction progress.
    """

    def setupUi(self, PredictWorkingDlg):
        PredictWorkingDlg.setObjectName("PredictWorkingDlg")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(definitions.PITECA_ICON_PATH), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        PredictWorkingDlg.setWindowIcon(icon)
        PredictWorkingDlg.resize(constants.DLG_DEFAULT_WIDTH, 64)
        self.verticalLayout = QtWidgets.QVBoxLayout(PredictWorkingDlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.progressBar = QtWidgets.QProgressBar(PredictWorkingDlg)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.label = QtWidgets.QLabel(PredictWorkingDlg)
        self.label.setText("Work in progress...")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)

        self.retranslateUi(PredictWorkingDlg)
        QtCore.QMetaObject.connectSlotsByName(PredictWorkingDlg)

    def retranslateUi(self, PredictWorkingDlg):
        _translate = QtCore.QCoreApplication.translate
        PredictWorkingDlg.setWindowTitle(_translate("PredictWorkingDlg", "Prediction"))
