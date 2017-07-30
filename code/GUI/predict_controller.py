import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from sharedutils.constants import Contrasts, Tasks
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
            #TODO: consider editing scenarios


    def findCheckedContrasts(self):
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
                    checkedSweeps.append(Contrasts[child.text(0)])
            contrasts[Tasks[signal.text(0)]] = checkedSweeps
        return contrasts


    def onContClicked(self):
        self.ui.existingFeatsUi.update_ids_to_extract()
        self.ids_to_extract = self.ui.existingFeatsDlg.ids_to_extract
        self.ui.existingFeatsDlg.close()


    def onRunPredictClicked(self):
        inputFiles = self.ui.inputFilesLineEdit.text()
        outputDir = self.ui.outputDirLineEdit.text()
        contrasts = self.findCheckedContrasts()
        predictModel = PredictModel(inputFiles, outputDir, contrasts)
        existingFeatures = predictModel.findExistingFeatures()

        self.ui.existingFeatsDlg = QtWidgets.QDialog()
        self.ui.existingFeatsDlg.setWindowModality(Qt.ApplicationModal)
        self.ui.existingFeatsUi = UIExistFeats(existingFeatures)
        self.ui.existingFeatsUi.setupUi(self.ui.existingFeatsDlg)
        self.ui.existingFeatsUi.contButton.clicked.connect(self.onContClicked)
        self.existingFeatsDlg.show()


        predictModel.extractFeatures(existingFeatures)
        predictModel.predict()
        return