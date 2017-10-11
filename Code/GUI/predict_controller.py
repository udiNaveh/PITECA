from GUI.settings_controller import get_prediction_outputs_folder

from PyQt5 import QtCore

from GUI.predict_model import PredictTabModel
from sharedutils.constants import Domain, Task
from sharedutils import dialog_utils
from sharedutils.constants import *
import definitions


class PredictController:
    """
    Connects user action on GUI to the prediction flow logic in the prediction model.
    """

    def __init__(self, ui):
        self.ui = ui

    def onBrowseInputFilesClicked(self):
        dir = definitions.ROOT_DIR
        dialog_utils.browse_files(self.ui.inputFilesLineEdit, dir)

    def findCheckedTasks(self):
        """
        Returns a list Task(s), that are checked in the UI.
        """

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

    def onRunPredictClicked(self):
        inputFiles = self.ui.inputFilesLineEdit.text()
        outputDir = get_prediction_outputs_folder()
        tasks = self.findCheckedTasks()
        if not inputFiles:
            dialog_utils.print_error(PROVIDE_INPUT_FILES_MSG)
            return
        if not tasks:
            dialog_utils.print_error(SELECT_TASKS_MSG)
            return
        predictModel = PredictTabModel(inputFiles, outputDir, tasks)
        predictModel.run_prediction_flow(self.ui)
