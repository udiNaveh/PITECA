from GUI.popups.dialogs import QuestionDialog
from PyQt5 import QtWidgets

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

    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.showMessage(msg)
    error_dialog.exec_()
