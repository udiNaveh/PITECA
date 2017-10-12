# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Sadna\src\git\PITECA\UI\results-dialog.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import definitions
from sharedutils import constants

class Ui_ResultsDialog(object):
    """
    Defines the UI view of the dialog shown after a CIFTI file/s were created during analysis of prediction.
    """

    def setupUi(self, ResultsDialog):
        ResultsDialog.setObjectName("ResultsDialog")
        ResultsDialog.resize(297, 112)
        ResultsDialog.setWindowIcon(QtGui.QIcon(definitions.PITECA_ICON_PATH))
        self.gridLayout = QtWidgets.QGridLayout(ResultsDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.openInFolderButton = QtWidgets.QPushButton(ResultsDialog)
        self.openInFolderButton.setMinimumSize(QtCore.QSize(90, 0))
        self.openInFolderButton.setMaximumSize(QtCore.QSize(90, 16777215))
        self.openInFolderButton.setObjectName("openInFolderButton")
        self.gridLayout.addWidget(self.openInFolderButton, 1, 1, 1, 1)
        self.okButton = QtWidgets.QPushButton(ResultsDialog)
        self.okButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.okButton.setObjectName("okButton")
        self.gridLayout.addWidget(self.okButton, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 3, 1, 1)
        self.msgLabel = QtWidgets.QLabel(ResultsDialog)
        self.msgLabel.setText("")
        self.msgLabel.setObjectName("msgLabel")
        self.gridLayout.addWidget(self.msgLabel, 0, 0, 1, 4)
        self.viewInWbButton = QtWidgets.QPushButton(ResultsDialog)
        self.viewInWbButton.setEnabled(False)
        self.viewInWbButton.setMinimumSize(QtCore.QSize(90, 0))
        self.viewInWbButton.setMaximumSize(QtCore.QSize(90, 16777215))
        self.viewInWbButton.setObjectName("viewInWbButton")
        self.gridLayout.addWidget(self.viewInWbButton, 1, 2, 1, 1)

        self.retranslateUi(ResultsDialog)
        QtCore.QMetaObject.connectSlotsByName(ResultsDialog)

    def retranslateUi(self, ResultsDialog):
        _translate = QtCore.QCoreApplication.translate
        ResultsDialog.setWindowTitle(_translate("ResultsDialog", "Results"))
        self.openInFolderButton.setText(_translate("ResultsDialog", "Open in Folder"))
        self.okButton.setText(_translate("ResultsDialog", constants.RESULTS_NEXT_BUTTON_TEXT))
        self.viewInWbButton.setText(_translate("ResultsDialog", "View in wb"))


