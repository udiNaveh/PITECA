import os
from PyQt5 import QtWidgets


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
