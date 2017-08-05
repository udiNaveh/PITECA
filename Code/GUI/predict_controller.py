import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from sharedutils.constants import Domain, Task
from GUI.predict_model import PredictModel
from misc.existingFeatsDlg import UIExistFeats

class PredictController:

    def __init__(self, ui):
        self.ui = ui


    def onBrowseInputFilesClicked(self):
        dlg = QtWidgets.QFileDialog()
        filters = "CIFTI (*.dtseries.nii)"
        files = dlg.getOpenFileNames(None, 'Select files', os.getcwd(), filters)[0]
        if files:
            self.ui.inputFilesLineEdit.setText(str(files))
            # TODO: consider editing scenarios


    def onBrowseOutputDirClicked(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory()
        if dir:
            self.ui.outputDirLineEdit.setText(dir)
            # TODO: consider editing scenarios


    def findCheckedTasks(self):
        contrasts = dict()
        root = self.ui.tasksTree.invisibleRootItem()
        signalCount = root.childCount()

        for i in range(signalCount):
            signal = root.child(i)
            checkedSweeps = list()
            numChildren = signal.childCount()

            for n in range(numChildren):
                child = signal.child(n)

                if child.checkState(0) == QtCore.Qt.Checked:
                    checkedSweeps.append(Domain[child.text(0)])
            contrasts[Task[signal.text(0)]] = checkedSweeps
        return contrasts


    def onContClicked(self):
        self.ui.existingFeatsUi.update_ids_to_extract()
        self.ids_to_extract = self.ui.existingFeatsDlg.ids_to_extract
        self.ui.existingFeatsDlg.close()


    def onRunPredictClicked(self):
        inputFiles = self.ui.inputFilesLineEdit.text()
        outputDir = self.ui.outputDirLineEdit.text()
        tasks = self.findCheckedTasks()
        predictModel = PredictModel(inputFiles, outputDir, tasks)
        predictModel.predict()
