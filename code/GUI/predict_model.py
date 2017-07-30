from PyQt5.QtCore import QFile, QIODevice
from sharedutils.constants import *
from sharedutils.string_utils import get_id, get_fetures_path
import os

class PredictModel:
    def __init__(self, inputFilesString, outputDir, contrasts):
        self.inputFiles = inputFilesString[1:].split(',')
        print(self.inputFiles)
        self.outputDir = outputDir
        self.contrasts = contrasts

    def findExistingFeatures(self):
        f = QFile(CONFIG_PATH)
        f.open(QIODevice.ReadOnly)
        featuresDir = f.readAll()
        f.close()

        existFeatsInpuFiles = []
        for fileName in self.inputFiles:
            print(1)
            id = get_id(fileName)
            print(fileName)
            path = get_fetures_path(id)
            if os.path.isfile(path):
                existFeatsInpuFiles.append(fileName)

        return existFeatsInpuFiles

    def extractFeatures(self, existingFeatures):
        for file in self.inputFiles:
            if file not in existingFeatures:
                print(file)

    def predict(self):
        return



