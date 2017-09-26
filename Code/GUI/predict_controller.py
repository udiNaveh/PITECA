import os
from GUI.settings_controller import get_prediction_outputs_folder

from PyQt5 import QtWidgets, QtCore

from GUI.predict_model import PredictTabModel
from sharedutils.constants import Domain, Task # TODO: change this to import constants
from sharedutils import general_utils, dialog_utils


class PredictController:

    def __init__(self, ui):
        self.ui = ui

    def onBrowseInputFilesClicked(self):
        dialog_utils.browse_files(self.ui.inputFilesLineEdit)

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
                    checkedSweeps.append(Task[child.text(0)])
            contrasts[Domain[signal.text(0)]] = checkedSweeps

        res = []
        for task in contrasts.values():
            for contrast in task:
                res.append(contrast)
        return res

    def onContClicked(self):
        self.ui.existingFeatsUi.update_ids_to_extract()
        self.ids_to_extract = self.ui.existingFeatsDlg.ids_to_extract
        self.ui.existingFeatsDlg.close()

    def onRunPredictClicked(self):
        inputFiles = self.ui.inputFilesLineEdit.text()
        outputDir = get_prediction_outputs_folder()
        tasks = self.findCheckedTasks()
        if (not inputFiles or not tasks):
            dialog_utils.print_error("Please provide input.")
            return
        predictModel = PredictTabModel(inputFiles, outputDir, tasks)
        predictModel.run_prediction_flow(self.ui)
