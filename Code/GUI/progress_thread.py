from PyQt5.QtCore import QThread, QObject
from PyQt5 import QtCore
import time, math


class ProgressThread(QThread):

    progress_update_sig = QtCore.pyqtSignal()

    def __init__(self, model, subjects, progress_bar, parent=None):
        super(ProgressThread, self).__init__(parent)
        self.model = model
        self.subjects = subjects
        self.progress_bar = progress_bar
        self.progress_update_sig.connect(self.update_progress_bar)

    def __del__(self):
        self.wait()

    def update_progress_bar(self):
        self.progress_bar.setValue(min(self.progress_bar.value() + math.ceil(100/len(self.subjects)), 100))

    def run(self):
        for subject in self.subjects:
            self.model.predict(subject)
            self.progress_update_sig.emit()
            # Tell the thread to sleep for 1 second and let other things run
            time.sleep(1)
        self.quit()