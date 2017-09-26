from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio
from textwrap import wrap

from sharedutils import dialog_utils, constants
from GUI.analyze_working_thread import AnalysisTask

"""
This module provides plotting methods of 4 heatmap graphs:
1. Correlation between the predicted activations of all subjects (subj_subj_data)
2. Correlation between the predicted activations of all subjects to their mean (subj_mean_data)
3. Correlation between the predicted activations of all subjects to the canonical activation of HCP subjects (subj_canonical_data)
4. Correlation between the predicted activations of all subjects to their actual activation (subj_subj_data)
"""

SUBJECTS_X_LABEL = "subjects"
SUBJECTS_Y_LABEL = "subjects"
MEAN_Y_LABEL = "with mean"
CANONICAL_Y_LABEL = "with canonical"

class GraphicDlg(QDialog):
    def __init__(self, analysis_task, data, subjects, title, parent=None):
        super(GraphicDlg, self).__init__(parent)

        # a figure instance to plot on
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # save button
        self.save_button = QtWidgets.QPushButton('Save data')
        self.save_button.setMaximumWidth(100)
        self.save_button.clicked.connect(self.save_data)

        # set the layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.save_button)
        self.setLayout(self.layout)

        # Add results illustration
        self.ids = [subject.subject_id for subject in subjects]
        self.analysis_task = analysis_task
        self.data = data
        self.title = title
        if self.analysis_task == AnalysisTask.Analysis_Correlations:
            self.subj_subj_data = data[0] # 2 dims
            self.subj_mean_data = data[1] # 1 dim
            self.subj_canonical_data = data[2] # 1 dim
            self.plot_heatmap()
        elif analysis_task == AnalysisTask.Compare_Correlations:
            self.subj_subj_data = data # 2 dims
            self.plot_heatmap()
        elif analysis_task == AnalysisTask.Compare_Significance:
            # self.data is 2 dimensional array
            self.plot_barchart()
        else:
            dialog_utils.print_error("Unsupported analysis action")
            return

        if self.analysis_task in [AnalysisTask.Analysis_Correlations, AnalysisTask.Compare_Correlations]:
            self.plot_heatmap()

    def onClick(self, event):
        if event.button == 1 and event.xdata and event.ydata:
            subject_x_index = math.floor(event.xdata)
            subject_y_index = math.floor(event.ydata)

            if self.analysis_task == AnalysisTask.Analysis_Correlations:
                # graph (1)
                if event.inaxes.get_xlabel() == SUBJECTS_X_LABEL and event.inaxes.get_ylabel() == SUBJECTS_Y_LABEL:
                    correlation = self.subj_subj_data[subject_x_index, subject_y_index]
                    between1 = "subject {}".format(self.ids[subject_x_index])
                    between2 = "subject {}".format(self.ids[subject_y_index])
                # graph (2)
                elif event.inaxes.get_xlabel() == "" and event.inaxes.get_ylabel() == MEAN_Y_LABEL:
                    correlation = self.subj_mean_data[subject_x_index]
                    between1 = self.ids[subject_x_index]
                    between2 = "mean activation"
                # graph (3)
                elif event.inaxes.get_xlabel() == "" and event.inaxes.get_ylabel() == CANONICAL_Y_LABEL:
                    correlation = self.subj_canonical_data[subject_x_index]
                    between1 = "subject {}".format(self.ids[subject_x_index])
                    between2 = "canonical activation"
                # not a heat map location
                else:
                    return

            elif self.analysis_task == AnalysisTask.Compare_Correlations:
                # graph 4
                if event.inaxes.get_xlabel() == SUBJECTS_X_LABEL and event.inaxes.get_ylabel() == SUBJECTS_Y_LABEL:
                    correlation = self.subj_subj_data[subject_x_index, subject_y_index]
                    between1 = "subject {}".format(self.ids[subject_x_index])
                    between2 = "subject {}".format(self.ids[subject_y_index])
                # not a heat map location
                else:
                    return

            else:
                dialog_utils.print_error(constants.UNEXPECTED_EXCEPTION_MSG)

            self.correlation_label.setText("Value: {:.2f} (Correlation between {} and {})"
                                           .format(correlation, between1, between2))

    def save_data(self):
        name, extension = dialog_utils.save_file('*.mat ;; *.numpy')
        if name == '':
            return
        extension = extension.split('.')[1]

        if extension == 'mat':
            sio.savemat(name, {'data': self.data})
        elif extension == 'numpy':
            self.data.tofile(name)
        else:
            raise Exception('File extension is not supported.')

    def plot_heatmap(self):

        # a label to show correlation
        self.correlation_label = QtWidgets.QLabel('Click on entry to see the exact correlation value')
        self.layout.addWidget(self.correlation_label)

        self.figure.clear()
        self.figure.suptitle(self.title)

        cmap = 'RdBu'
        edgecolors = 'black'

        # calculate x tick labels font
        num_of_chars = len(''.join(self.ids)) + len(self.ids)
        font_size = (14 / (math.ceil(num_of_chars / 29))) if num_of_chars > 29 else 7

        # create an axis
        subjects_by_subjects_ax = self.figure.gca()
        subjects_by_subjects_ax.set_aspect('equal', adjustable='box')
        subjects_by_subjects_ax.set_xlabel(SUBJECTS_X_LABEL)
        subjects_by_subjects_ax.set_ylabel(SUBJECTS_Y_LABEL)
        subjects_by_subjects_ax.set_xticks(np.arange(len(self.subj_subj_data)) + 0.5)
        subjects_by_subjects_ax.set_xticklabels(self.ids, fontsize=font_size, rotation=50)
        subjects_by_subjects_ax.set_yticks(np.arange(len(self.subj_subj_data)) + 0.5)
        subjects_by_subjects_ax.set_yticklabels(self.ids, fontsize=font_size, rotation=50)

        # plot data
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(subjects_by_subjects_ax)
        color_ax = divider.append_axes("right", "5%", pad="5%")

        if self.analysis_task == AnalysisTask.Analysis_Correlations:
            heatmap_s_s = subjects_by_subjects_ax.pcolor(self.subj_subj_data, cmap=cmap, vmin=-1, vmax=1, edgecolors=edgecolors)
            # add correlations to mean and canonical
            subjects_by_mean_ax = divider.append_axes("bottom", "7%", pad="60%")
            subjects_by_canonical_ax = divider.append_axes("bottom", "7%", pad="50%")

            self.figure.add_axes(subjects_by_mean_ax)
            self.figure.add_axes(subjects_by_canonical_ax)
            subjects_by_mean_ax.pcolor([self.subj_mean_data], cmap=cmap, vmin=-1, vmax=1, edgecolors=edgecolors)
            subjects_by_canonical_ax.pcolor([self.subj_canonical_data], cmap=cmap, vmin=-1, vmax=1, edgecolors=edgecolors)

            subjects_by_mean_ax.set_ylabel(MEAN_Y_LABEL, rotation=0)
            subjects_by_mean_ax.yaxis.set_label_coords(-0.5, 0)
            subjects_by_mean_ax.set_xticks(np.arange(len(self.subj_mean_data)) + 0.5)
            subjects_by_mean_ax.set_xticklabels(self.ids, fontsize=font_size, rotation=50)
            subjects_by_mean_ax.set_yticks(np.arange(0))

            subjects_by_canonical_ax.set_ylabel(CANONICAL_Y_LABEL, rotation=0)
            subjects_by_canonical_ax.yaxis.set_label_coords(-0.5, 0)
            subjects_by_canonical_ax.set_xticks(np.arange(len(self.subj_canonical_data)) + 0.5)
            subjects_by_canonical_ax.set_xticklabels(self.ids, fontsize=font_size, rotation=50)
            subjects_by_canonical_ax.set_yticks(np.arange(0))

        else:
            heatmap_s_s = subjects_by_subjects_ax.pcolor(self.data, cmap=cmap, vmin=-1, vmax=1, edgecolors=edgecolors)

        self.figure.colorbar(heatmap_s_s, cax=color_ax)
        self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', self.onClick)
        plt.tight_layout()

    def plot_barchart(self):

        self.figure.clear()
        ax = self.figure.gca()

        men_means = (20, 35, 30, 35, 27)
        N = 5

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        rects1 = ax.bar(ind, men_means, width, color='steelblue', yerr=men_means)

        women_means = (25, 32, 34, 20, 25)
        women_std = (3, 5, 2, 3, 3)
        rects2 = ax.bar(ind + width, women_means, width, color='mediumspringgreen', yerr=women_std)

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

        ax.legend((rects1[0], rects2[0]), ('Predicted', 'Actual'))

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        # refresh canvas
        self.canvas.draw()



