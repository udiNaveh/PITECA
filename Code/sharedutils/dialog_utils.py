from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import Qt, QSize

from GUI.popups.question_dlg import QuestionDialog
from GUI.popups import results_dlg_controller, results_dlg_view
import definitions

'''
This utility enables popup from any part of the code. 
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
    error_dialog.setBaseSize(QSize(10000000, 120))
    error_dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    error_dialog.setSizeGripEnabled(True)
    error_dialog.setWindowModality(Qt.ApplicationModal)
    error_dialog.setWindowIcon(QtGui.QIcon(definitions.PITECA_ICON_PATH))
    error_dialog.setWindowTitle("Error")
    error_dialog.setText(msg)
    error_dialog.setMinimumWidth(10000000)
    error_dialog.addButton(QtWidgets.QMessageBox.Ok)
    error_dialog.exec_()

def inform_user(msg):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setBaseSize(QSize(10000000, 120))
    msg_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    msg_box.setSizeGripEnabled(True)
    msg_box.setWindowModality(Qt.ApplicationModal)
    msg_box.setWindowIcon(QtGui.QIcon(definitions.PITECA_ICON_PATH))
    msg_box.setWindowTitle("Message")
    msg_box.setMinimumWidth(10000000)
    msg_box.setText(msg)
    msg_box.addButton(QtWidgets.QMessageBox.Ok)
    msg_box.exec_()


def report_results(msg, folder, filepath, exit_on_open_folder=True):
    """
    This function pops up a dialog that informs the user about a CIFTI files
    that were created by PITECA and saved in the computer.
    :param msg: The string message to show in the dialog.
    :param folder: The string path of the folder to be opened when "Open in folder" is clicked.
    :param filepath: The string path of the file that was created. Provided for situations where only one file
    was created and the "Open in wb" is enabled.
    :param exit_on_open_folder: a boolean argument defines if dialog should be closed after "Open in..." buttons.
    """
    dlg = results_dlg_controller.ResultsDlg(msg, folder, exit_on_open_folder, filepath)
    ui = results_dlg_view.Ui_ResultsDialog()
    ui.setupUi(dlg)
    dlg.update_ui(ui)
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.show()
    dlg.exec_()


def browse_files(line_edit, dir):
    """
    A function to open browse dialog to select files.
    :param line_edit: the UI element where the selected file paths shown.
    :param dir: The directory the browser should be opened at.
    """
    dlg = QtWidgets.QFileDialog()
    filters = "CIFTI (*.dtseries.nii)"
    files = dlg.getOpenFileNames(None, 'Select Files', dir, filters)[0]
    if files:
        line_edit.setText(str(files))


def browse_dir(line_edit, dir):
    """
    A function to open browse dialog to select a folder.
    :param line_edit: the UI element where the selected folder path shown.
    :param dir: The directory the browser should be opened at.
    """
    dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder', dir)
    if dir:
        line_edit.setText(dir)
    return dir


def save_file(filters, dir):
    return QtWidgets.QFileDialog.getSaveFileName(None, 'Select Files', dir, filters)