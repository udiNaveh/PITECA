from PyQt5.QtWidgets import QWidget, QMessageBox, QMainWindow, QLabel, QGridLayout, QDesktopWidget
from PyQt5 import QtWidgets, QtGui

import definitions


width = 320
height = 200

class QuestionDialog(QWidget):
    '''
    Includes view + controller of a generic dialog shown to get an yes or no answer from user.
    '''

    def __init__(self, default_ans, title, msg):
        super().__init__()
        self.title = title
        self.msg = msg
        self.width = width
        self.height = height
        fg = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())
        self.ans = default_ans
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(definitions.PITECA_ICON_PATH), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.initUI()

    def initUI(self):

        buttonReply = QMessageBox.question(self, self.title, self.msg,
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.ans = True
        self.show()


