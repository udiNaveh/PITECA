from PyQt5.QtWidgets import QWidget, QMessageBox, QMainWindow, QLabel, QGridLayout, QDesktopWidget
from PyQt5 import QtWidgets



class QuestionDialog(QWidget):
    '''
    Includes view + controller
    '''

    def __init__(self, default_ans, title, msg):
        super().__init__()
        self.title = title
        self.msg = msg
        self.width = 320
        self.height = 200
        fg = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())
        self.ans = default_ans
        self.initUI()

    def initUI(self):

        buttonReply = QMessageBox.question(self, self.title, self.msg,
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.ans = True
        self.show()


