from GUI.popups.question_dlg import QuestionDialog
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
import definitions
import os

'''
This utility enables popup from any part of the code. 
For now we have only one type of dialog: a question dialog (user should choose Yes or No)
'''


def ask_user(default_ans, title, question):
    '''
    This function pops up a question dialog for the user.
    The window that pops contains question mark icon, a question, and Yes and No buttons.
    :param default_ans: The answer to consider if user pushed the X button
    :param title: The title of the window
    :param question: The message shown
    :return: A boolean value specifies if user pushed The Yes button
    '''
    dlg = QuestionDialog(default_ans, title, question)
    return dlg.ans

def print_error(msg):
    '''
    This function pops up an error message for the user.
    The user can click OK or X to close the window.
    :param msg: The error to be printed for the user.
    '''

    error_dialog = QtWidgets.QMessageBox()
    error_dialog.setWindowModality(Qt.ApplicationModal)
    error_dialog.setWindowIcon(QtGui.QIcon(definitions.PITECA_ICON_PATH))
    error_dialog.setWindowTitle("Error")
    error_dialog.setText(msg)
    error_dialog.setFixedWidth(10000000)
    error_dialog.addButton(QtWidgets.QMessageBox.Ok)
    error_dialog.exec_()

def inform_user(msg):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowModality(Qt.ApplicationModal)
    msg_box.setWindowIcon(QtGui.QIcon(definitions.PITECA_ICON_PATH))
    msg_box.setWindowTitle("Message")
    msg_box.setMinimumSize(10000000, 0)
    msg_box.setText(msg)
    msg_box.addButton(QtWidgets.QMessageBox.Ok)
    msg_box.exec_()

def browse_files(line_edit):
    dlg = QtWidgets.QFileDialog()
    filters = "CIFTI (*.dtseries.nii)"
    files = dlg.getOpenFileNames(None, 'Select files', os.getcwd(), filters)[0]
    if files:
        line_edit.setText(str(files))
        # TODO: consider editing scenarios


def browse_dir(line_edit):
    dir = QtWidgets.QFileDialog.getExistingDirectory()
    if dir:
        line_edit.setText(dir)
        # TODO: consider editing scenarios
    return dir


def save_file(filters):
    return QtWidgets.QFileDialog.getSaveFileName(None, 'Select files', os.getcwd(), filters)