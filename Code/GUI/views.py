# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Sadna\src\git\PITECA\UI\main-window-responsive.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from sharedutils.constants import Domain, AVAILABLE_TASKS
from definitions import PITECA_ICON_PATH

class Ui_MainView(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(453, 573)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(PITECA_ICON_PATH), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(9, -1, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mainView = QtWidgets.QTabWidget(self.centralwidget)
        self.mainView.setObjectName("mainView")
        self.predictView = QtWidgets.QWidget()
        self.predictView.setObjectName("predictView")
        self.gridLayout = QtWidgets.QGridLayout(self.predictView)
        self.gridLayout.setContentsMargins(9, 9, 9, 9)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 5, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 5, 3, 1, 1)
        self.browseInputFilesButton = QtWidgets.QPushButton(self.predictView)
        self.browseInputFilesButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.browseInputFilesButton.setObjectName("browseInputFilesButton")
        self.gridLayout.addWidget(self.browseInputFilesButton, 2, 3, 1, 1)
        self.inputFilesLineEdit = QtWidgets.QLineEdit(self.predictView)
        self.inputFilesLineEdit.setObjectName("inputFilesLineEdit")
        self.gridLayout.addWidget(self.inputFilesLineEdit, 2, 0, 1, 3)
        self.runPredictButton = QtWidgets.QPushButton(self.predictView)
        self.runPredictButton.setObjectName("runPredictButton")
        self.gridLayout.addWidget(self.runPredictButton, 5, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 5, 2, 1, 1)
        self.tasksTree = QtWidgets.QTreeWidget(self.predictView)
        self.tasksTree.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tasksTree.setColumnCount(1)
        self.tasksTree.setObjectName("tasksTree")
        self.tasksTree.headerItem().setText(0, "1")
        self.gridLayout.addWidget(self.tasksTree, 4, 0, 1, 2)
        self.selectInputFilesLabel = QtWidgets.QLabel(self.predictView)
        self.selectInputFilesLabel.setObjectName("selectInputFilesLabel")
        self.gridLayout.addWidget(self.selectInputFilesLabel, 1, 0, 1, 2)
        self.mainView.addTab(self.predictView, "")
        self.analyzeView = QtWidgets.QWidget()
        self.analyzeView.setObjectName("analyzeView")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.analyzeView)
        self.gridLayout_2.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 11, 1, 1, 1)
        self.selectPredictedLabel = QtWidgets.QLabel(self.analyzeView)
        self.selectPredictedLabel.setObjectName("selectPredictedLabel")
        self.gridLayout_2.addWidget(self.selectPredictedLabel, 3, 0, 1, 2)
        self.analysisCorrelationsRadioButton = QtWidgets.QRadioButton(self.analyzeView)
        self.analysisCorrelationsRadioButton.setObjectName("analysisCorrelationsRadioButton")
        self.gridLayout_2.addWidget(self.analysisCorrelationsRadioButton, 9, 0, 1, 2)
        # self.analysisSignificantRadioButton = QtWidgets.QRadioButton(self.analyzeView)
        # self.analysisSignificantRadioButton.setObjectName("analysisSignificantRadioButton")
        # self.gridLayout_2.addWidget(self.analysisSignificantRadioButton, 10, 0, 1, 2)
        self.analysisMeanRadioButton = QtWidgets.QRadioButton(self.analyzeView)
        self.analysisMeanRadioButton.setObjectName("analysisMeanRadioButton")
        self.gridLayout_2.addWidget(self.analysisMeanRadioButton, 8, 0, 1, 2)
        self.domainComboBox = QtWidgets.QComboBox(self.analyzeView)
        self.domainComboBox.setObjectName("domainComboBox")
        self.gridLayout_2.addWidget(self.domainComboBox, 1, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 2, 0, 1, 1)
        self.addActualLineEdit = QtWidgets.QLineEdit(self.analyzeView)
        self.addActualLineEdit.setObjectName("addActualLineEdit")
        self.gridLayout_2.addWidget(self.addActualLineEdit, 14, 0, 1, 4)
        self.runCompareButton = QtWidgets.QPushButton(self.analyzeView)
        self.runCompareButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.runCompareButton.setObjectName("runCompareButton")
        self.gridLayout_2.addWidget(self.runCompareButton, 17, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem5, 18, 0, 1, 1)
        self.runAnalyzeButton = QtWidgets.QPushButton(self.analyzeView)
        self.runAnalyzeButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.runAnalyzeButton.setObjectName("runAnalyzeButton")
        self.gridLayout_2.addWidget(self.runAnalyzeButton, 11, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem6, 11, 2, 1, 1)
        self.browsePredictedButton = QtWidgets.QPushButton(self.analyzeView)
        self.browsePredictedButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.browsePredictedButton.setObjectName("browsePredictedButton")
        self.gridLayout_2.addWidget(self.browsePredictedButton, 7, 4, 1, 1)
        self.taskComboBox = QtWidgets.QComboBox(self.analyzeView)
        self.taskComboBox.setObjectName("taskComboBox")
        self.gridLayout_2.addWidget(self.taskComboBox, 1, 1, 1, 1)
        self.browseActualButton = QtWidgets.QPushButton(self.analyzeView)
        self.browseActualButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.browseActualButton.setObjectName("browseActualButton")
        self.gridLayout_2.addWidget(self.browseActualButton, 14, 4, 1, 1)
        self.selectPredictedLineEdit = QtWidgets.QLineEdit(self.analyzeView)
        self.selectPredictedLineEdit.setObjectName("selectPredictedLineEdit")
        self.gridLayout_2.addWidget(self.selectPredictedLineEdit, 7, 0, 1, 4)
        self.addActualLabel = QtWidgets.QLabel(self.analyzeView)
        self.addActualLabel.setObjectName("addActualLabel")
        self.gridLayout_2.addWidget(self.addActualLabel, 13, 0, 1, 2)
        self.selectDomainTaskLabel = QtWidgets.QLabel(self.analyzeView)
        self.selectDomainTaskLabel.setObjectName("selectDomainTaskLabel")
        self.gridLayout_2.addWidget(self.selectDomainTaskLabel, 0, 0, 1, 2)
        self.comparisonCorrelationsRadioButton = QtWidgets.QRadioButton(self.analyzeView)
        self.comparisonCorrelationsRadioButton.setObjectName("comparisonCorrelationsRadioButton")
        self.gridLayout_2.addWidget(self.comparisonCorrelationsRadioButton, 15, 0, 1, 2)
        self.comparisonSignificantRadioButton = QtWidgets.QRadioButton(self.analyzeView)
        self.comparisonSignificantRadioButton.setObjectName("comparisonSignificantRadioButton")
        self.gridLayout_2.addWidget(self.comparisonSignificantRadioButton, 16, 0, 1, 2)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem7, 12, 0, 1, 1)
        self.mainView.addTab(self.analyzeView, "")
        self.settingsView = QtWidgets.QWidget()
        self.settingsView.setObjectName("settingsView")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.settingsView)
        self.gridLayout_3.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.predictionOutputFolderButton = QtWidgets.QPushButton(self.settingsView)
        self.predictionOutputFolderButton.setObjectName("predictionOutputFolderButton")
        self.gridLayout_3.addWidget(self.predictionOutputFolderButton, 3, 1, 1, 1)
        self.featuresFolderButton = QtWidgets.QPushButton(self.settingsView)
        self.featuresFolderButton.setObjectName("featuresFolderButton")
        self.gridLayout_3.addWidget(self.featuresFolderButton, 1, 1, 1, 1)
        self.analysisOutputFolderButton = QtWidgets.QPushButton(self.settingsView)
        self.analysisOutputFolderButton.setObjectName("analysisOutputFolderButton")
        self.gridLayout_3.addWidget(self.analysisOutputFolderButton, 5, 1, 1, 1)
        self.featuresFolderLineEdit = QtWidgets.QLineEdit(self.settingsView)
        self.featuresFolderLineEdit.setObjectName("featuresFolderLineEdit")
        self.gridLayout_3.addWidget(self.featuresFolderLineEdit, 1, 0, 1, 1)
        self.predictionOutputFolderLabel = QtWidgets.QLabel(self.settingsView)
        self.predictionOutputFolderLabel.setObjectName("predictionOutputFolderLabel")
        self.gridLayout_3.addWidget(self.predictionOutputFolderLabel, 2, 0, 1, 1)
        self.featuresFolderLabel = QtWidgets.QLabel(self.settingsView)
        self.featuresFolderLabel.setObjectName("featuresFolderLabel")
        self.gridLayout_3.addWidget(self.featuresFolderLabel, 0, 0, 1, 1)
        self.predictionOutputFolderLineEdit = QtWidgets.QLineEdit(self.settingsView)
        self.predictionOutputFolderLineEdit.setObjectName("predictionOutputFolderLineEdit")
        self.gridLayout_3.addWidget(self.predictionOutputFolderLineEdit, 3, 0, 1, 1)
        self.analysisOutputFolderLabel = QtWidgets.QLabel(self.settingsView)
        self.analysisOutputFolderLabel.setObjectName("analysisOutputFolderLabel")
        self.gridLayout_3.addWidget(self.analysisOutputFolderLabel, 4, 0, 1, 1)
        self.analysisOutputFolderLineEdit = QtWidgets.QLineEdit(self.settingsView)
        self.analysisOutputFolderLineEdit.setObjectName("analysisOutputFolderLineEdit")
        self.gridLayout_3.addWidget(self.analysisOutputFolderLineEdit, 5, 0, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem8, 6, 0, 1, 1)
        self.mainView.addTab(self.settingsView, "")
        self.horizontalLayout.addWidget(self.mainView)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.mainView.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate

        """
        Added manually
        """
        self.tasksTree.headerItem().setText(0, _translate("MainView", "Tasks"))
        for domain in Domain:
            taskItem = QtWidgets.QTreeWidgetItem(self.tasksTree)
            taskItem.setText(0, domain.name)
            for task in domain.value:
                if task in AVAILABLE_TASKS:
                    contrastItem = QtWidgets.QTreeWidgetItem(taskItem)
                    contrastItem.setCheckState(0, QtCore.Qt.Checked)
                    contrastItem.setCheckState(0, QtCore.Qt.Unchecked)
                    contrastItem.setText(0, task.name)
            self.domainComboBox.addItem(domain.name)
        self.taskComboBox.addItems(task.name for task in Domain.EMOTION.value)
        """ End of Added manually """

        MainWindow.setWindowTitle(_translate("MainWindow", "PITECA"))
        self.browseInputFilesButton.setText(_translate("MainWindow", "Browse..."))
        self.runPredictButton.setText(_translate("MainWindow", "Run"))
        self.selectInputFilesLabel.setText(_translate("MainWindow", "Select input files:"))
        self.mainView.setTabText(self.mainView.indexOf(self.predictView), _translate("MainWindow", "Predict"))
        self.selectPredictedLabel.setText(_translate("MainWindow", "Select predicted activation files:"))
        self.analysisCorrelationsRadioButton.setText(_translate("MainWindow", "Correlations"))
        # self.analysisSignificantRadioButton.setText(_translate("MainWindow", "Areas of significant activation"))
        self.analysisMeanRadioButton.setText(_translate("MainWindow", "Get mean predicted activation for group"))
        self.runCompareButton.setText(_translate("MainWindow", "Compare"))
        self.runAnalyzeButton.setText(_translate("MainWindow", "Analyze"))
        self.browsePredictedButton.setText(_translate("MainWindow", "Browse..."))
        self.browseActualButton.setText(_translate("MainWindow", "Browse..."))
        self.addActualLabel.setText(_translate("MainWindow", "Add actual activation files to compare:"))
        self.selectDomainTaskLabel.setText(_translate("MainWindow", "Select Domain and Task:"))
        self.comparisonCorrelationsRadioButton.setText(_translate("MainWindow", "Correlations"))
        self.comparisonSignificantRadioButton.setText(_translate("MainWindow", "Areas of significant activation"))
        self.mainView.setTabText(self.mainView.indexOf(self.analyzeView), _translate("MainWindow", "Analyze Results"))
        self.predictionOutputFolderButton.setText(_translate("MainWindow", "Browse..."))
        self.featuresFolderButton.setText(_translate("MainWindow", "Browse..."))
        self.analysisOutputFolderButton.setText(_translate("MainWindow", "Browse..."))
        self.predictionOutputFolderLabel.setText(_translate("MainWindow", "Set output folder for prediction results:"))
        self.featuresFolderLabel.setText(_translate("MainWindow", "Set features folder:"))
        self.analysisOutputFolderLabel.setText(_translate("MainWindow", "Set output folder for analysis results:"))
        self.mainView.setTabText(self.mainView.indexOf(self.settingsView), _translate("MainWindow", "Settings"))
