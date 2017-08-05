# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\PycharmProjects\Sadna\GUI\ui\existing-features-dlg.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class UIExistFeats(object):

    def __init__(self, ids):
        self.ids = ids
        self.ids_to_extract = []

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(339, 285)
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 50, 256, 181))
        self.listWidget.setObjectName("listWidget")
        for id in self.ids:
            item = QtWidgets.QListWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsDragEnabled|QtCore.Qt.ItemIsUserCheckable|QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Checked)
            self.listWidget.addItem(item)
        self.contButton = QtWidgets.QPushButton(Dialog)
        self.contButton.setGeometry(QtCore.QRect(20, 240, 75, 23))
        self.contButton.setObjectName("contButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 10, 281, 31))
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        for i in range(len(self.ids)):
            item = self.listWidget.item(i)
            item.setText(_translate("Dialog", self.ids[i]))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.contButton.setText(_translate("Dialog", "Continue"))
        self.label.setText(_translate("Dialog", "Features for the following subjects already exist in: "))

    def update_ids_to_extract(self):
        for i in range(len(self.ids)):
            if self.listWidget.item(i).checkState() == 0:
                self.ids_to_extract.append(self.ids[i])



