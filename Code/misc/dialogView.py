# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Sadna\src\git\PITECA\Code\misc\dialog_feats.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(285, 246)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.yesButton = QtWidgets.QPushButton(Dialog)
        self.yesButton.setObjectName("yesButton")
        self.gridLayout.addWidget(self.yesButton, 1, 0, 1, 1)
        self.noButton = QtWidgets.QPushButton(Dialog)
        self.noButton.setObjectName("noButton")
        self.gridLayout.addWidget(self.noButton, 1, 1, 1, 1)
        self.dialogLabel = QtWidgets.QLabel(Dialog)
        self.dialogLabel.setObjectName("dialogLabel")
        self.gridLayout.addWidget(self.dialogLabel, 0, 0, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.yesButton.setText(_translate("Dialog", "Yes"))
        self.noButton.setText(_translate("Dialog", "No"))
        self.dialogLabel.setText(_translate("Dialog", "<html><head/><body><p>Some of the subjects seem to have extracted features.</p><p>Do you want to use the existing features?</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

