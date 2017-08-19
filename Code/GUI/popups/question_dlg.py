from PyQt5.QtWidgets import QWidget, QMessageBox, QMainWindow, QLabel, QGridLayout, QWidget, QDesktopWidget



class QuestionDialog(QWidget):
    '''
    Includes view + controller
    '''

    def __init__(self, default_ans, title, msg):
        super().__init__()
        self.title = title
        self.msg = msg
        self.left = 500
        self.top = 250
        self.width = 320
        self.height = 200
        # TODO: center the window (the commented code doesn't work as is)
        # qtRectangle = self.frameGeometry()
        # centerPoint = QDesktopWidget().availableGeometry().center()
        # qtRectangle.moveCenter(centerPoint)
        # self.move(qtRectangle.topLeft())
        self.ans = default_ans
        self.initUI()

    def initUI(self):
        self.setGeometry(self.left, self.top, self.width, self.height)

        buttonReply = QMessageBox.question(self, self.title, self.msg,
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.ans = True
        self.show()


