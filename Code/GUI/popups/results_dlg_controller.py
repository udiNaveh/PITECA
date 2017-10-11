import platform, os, subprocess

from PyQt5 import QtWidgets

from sharedutils import cmd_utils


class ResultsDlg(QtWidgets.QDialog):
    """
    The dialog shown to the user after a progress that created and saved new CIFTI files.
    """

    def __init__(self, label_text, folder, exit_on_open_folder, filepath):
        super(ResultsDlg, self).__init__()
        self.label_text = label_text
        self.folder = folder
        self.exit_on_open_folder = exit_on_open_folder
        self.filepath = filepath

    def update_ui(self, ui):
        """
        Some parameters are set only after the ui element is created.
        Before showing it, the ui has to be updated according to the new data.
        :param ui: the ui element to be updated (results dialog ui)
        """
        if self.filepath:
            ui.viewInWbButton.setEnabled(True)
        ui.msgLabel.setText(self.label_text)
        ui.openInFolderButton.clicked.connect(lambda: self.open_in_folder())
        ui.okButton.clicked.connect(lambda: self.close())
        ui.viewInWbButton.clicked.connect(lambda: self.open_in_wb())

    def open_in_folder(self):
        # Implementation if we do want to close this dialog after "open in folder":
        # if self.exit_on_open_folder:
        #     self.close()
        if platform.system() == "Windows":
            os.startfile(self.folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", self.folder])
        else:
            subprocess.Popen(["xdg-open", self.folder])

    def open_in_wb(self):
        """
        The function to be called when user clicks "open in wb" button.
        """
        cmd_utils.show_maps_in_wb_view(self.filepath)

