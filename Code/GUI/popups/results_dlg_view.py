# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Sadna\src\git\PITECA\UI\results-dialog.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from definitions import PITECA_ICON_PATH

class Ui_ResultsDialog(object):
    def setupUi(self, ResultsDialog):
        ResultsDialog.setObjectName("ResultsDialog")
        ResultsDialog.resize(297, 112)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(PITECA_ICON_PATH), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ResultsDialog.setWindowIcon(icon)
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
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        self.msgLabel = QtWidgets.QLabel(ResultsDialog)
        self.msgLabel.setText("")
        self.msgLabel.setObjectName("msgLabel")
        self.gridLayout.addWidget(self.msgLabel, 0, 0, 1, 3)

        self.retranslateUi(ResultsDialog)
        QtCore.QMetaObject.connectSlotsByName(ResultsDialog)

    def retranslateUi(self, ResultsDialog):
        _translate = QtCore.QCoreApplication.translate
        ResultsDialog.setWindowTitle(_translate("ResultsDialog", "Results"))
        self.openInFolderButton.setText(_translate("ResultsDialog", "Open in Folder"))
        self.okButton.setText(_translate("ResultsDialog", "OK"))
