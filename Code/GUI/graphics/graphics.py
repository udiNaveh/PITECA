from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio
import pickle

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from sharedutils import dialog_utils, constants, io_utils
from GUI.analyze_working_thread import AnalysisTask
import definitions

"""
This module provides plotting methods of 4 heatmap graphs:
1. Correlation between the predicted activations of all subjects (subj_subj_data)
2. Correlation between the predicted activations of all subjects to their mean (subj_mean_data)
3. Correlation between the predicted activations of all subjects to the canonical activation of HCP subjects (subj_canonical_data)
4. Correlation between the predicted activations of all subjects to their actual activation (subj_subj_data)
"""

# MEAN_Y_LABEL = "with mean"
CANONICAL_Y_LABEL = "correlation with\n canonical activation"

min_corr=0
max_corr=1

class GraphicDlg(QDialog):
    def __init__(self, analysis_task, data, subjects, title, parent=None):
        super(GraphicDlg, self).__init__(parent)

        # a figure instance to plot on
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
        self.setWindowTitle("Graph")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(definitions.PITECA_ICON_PATH), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        # Add results illustration
        self.ids = [subject.subject_id for subject in subjects]
        self.analysis_task = analysis_task
        self.data = data
        self.title = title
        self.named_data = {}

        # calculate x tick labels font
        num_of_chars = len(''.join(self.ids)) + len(self.ids)
        self.font_size = (14 / (math.ceil(num_of_chars / 29))) if num_of_chars > 29 else 7

        # Set graph attributes specifically by tasks
        if self.analysis_task == AnalysisTask.Analysis_Correlations:
            self.SUBJECTS_X_LABEL = "subjects"
            self.SUBJECTS_Y_LABEL = "subjects"
            self.subj_subj_data = data[0] # 2 dims
            self.subj_mean_data = data[1] # 1 dim
            self.subj_canonical_data = data[2] # 1 dim
            self.named_data = {'inter-subject predictions correlation' : self.subj_subj_data,
                               'subjects predictions correlations with canonical' : self.subj_canonical_data}
            self.plot_heatmap()
        elif analysis_task == AnalysisTask.Compare_Correlations:
            self.SUBJECTS_X_LABEL = "subjects: Actual"
            self.SUBJECTS_Y_LABEL = "subjects: Predicted"
            self.subj_subj_data = data # 2 dims
            self.named_data = {'inter-subject predicted-actual correlations' : self.subj_subj_data,
                               }
            self.plot_heatmap()
        if analysis_task == AnalysisTask.Analysis_Correlations or analysis_task == AnalysisTask.Compare_Correlations:
            if analysis_task == AnalysisTask.Compare_Correlations:
                mean_of_diagonal = np.mean(np.diagonal(self.subj_subj_data))
                self.mean_correlation_label = QtWidgets.QLabel('Mean correlation: {:01.2f}'.format(mean_of_diagonal))
                self.layout.addWidget(self.mean_correlation_label)
            # a label to show correlation
            self.correlation_label = QtWidgets.QLabel('Click on entry to see the exact correlation value')
            self.layout.addWidget(self.correlation_label)
        elif analysis_task == AnalysisTask.Compare_Significance:
            # self.data is 2 dimensional array
            self.named_data = {'subjects presicted-actual positive significance iou ': self.data[0],
                               'subjects presicted-actual negative significance iou ': self.data[1]}
            self.plot_barchart()
        else:
            dialog_utils.print_error("Unsupported analysis action")
            return

        if self.analysis_task in [AnalysisTask.Analysis_Correlations, AnalysisTask.Compare_Correlations]:
            self.plot_heatmap()

    def onClick(self, event):
        """
        The function that should be called when user clicks on the graph.
        Changes the label that shows the exact correlation if user clicked inside graph borders,
        or do nothing if user clicks elsewhere.
        """
        if event.button == 1 and event.xdata and event.ydata:
            subject_x_index = math.floor(event.xdata)
            subject_y_index = math.floor(event.ydata)

            if self.analysis_task == AnalysisTask.Analysis_Correlations:
                # graph (1)
                if event.inaxes.get_xlabel() == self.SUBJECTS_X_LABEL and event.inaxes.get_ylabel() == self.SUBJECTS_Y_LABEL:
                    correlation = self.subj_subj_data[subject_x_index, subject_y_index]
                    between1 = "subject {}".format(self.ids[subject_x_index])
                    between2 = "subject {}".format(self.ids[subject_y_index])
                # graph (2)
                # elif event.inaxes.get_xlabel() == "" and event.inaxes.get_ylabel() == MEAN_Y_LABEL:
                #     correlation = self.subj_mean_data[subject_x_index]
                #     between1 = self.ids[subject_x_index]
                #     between2 = "mean activation"
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
                if event.inaxes.get_xlabel() == self.SUBJECTS_X_LABEL and event.inaxes.get_ylabel() == self.SUBJECTS_Y_LABEL:
                    correlation = self.subj_subj_data[subject_y_index, subject_x_index]
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
        """
        The function that is called when user clicks on "save data" button
        Opens a browse dialog and saves the data shown in the graph in the format and folder
        the user selected.
        """
        dir = definitions.ANALYSIS_DIR
        name, extension = dialog_utils.save_file('*.mat ;; *.npy ;; *.pkl', dir)
        if name == '':
            return
        extension = extension.split('.')[1]

        if extension == 'mat':
            sio.savemat(name, {'data': self.data})
        elif extension == 'npy':
            self.ids
            np.save(name, np.asarray(self.data))
        elif extension == 'pkl':
            data_to_save = self.named_data
            data_to_save['subjects ids'] = self.ids
            io_utils.save_pickle(data_to_save, name)
        else:
            raise ValueError('File extension is not supported.')

    def plot_heatmap(self):
        """
        Plots a heatmap graph according to data
        """

        plt.gcf().subplots_adjust(bottom=0.2)

        self.figure.clear()
        self.figure.suptitle(self.title)

        cmap = 'Reds'
        edgecolors = 'black'

        # create an axis
        subjects_by_subjects_ax = self.figure.gca()
        subjects_by_subjects_ax.set_aspect('equal', adjustable='box')
        subjects_by_subjects_ax.set_xlabel(self.SUBJECTS_X_LABEL)
        subjects_by_subjects_ax.set_ylabel(self.SUBJECTS_Y_LABEL)
        subjects_by_subjects_ax.set_xticks(np.arange(len(self.subj_subj_data)) + 0.5)
        subjects_by_subjects_ax.set_xticklabels(self.ids, fontsize=self.font_size, rotation=35)
        subjects_by_subjects_ax.set_yticks(np.arange(len(self.subj_subj_data)) + 0.5)
        subjects_by_subjects_ax.set_yticklabels(self.ids, fontsize=self.font_size, rotation=35)

        plt.subplots_adjust(top=0.84)

        # plot data
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(subjects_by_subjects_ax)
        color_ax = divider.append_axes("right", "5%", pad="5%")

        if self.analysis_task == AnalysisTask.Analysis_Correlations:
            heatmap_s_s = subjects_by_subjects_ax.pcolor(self.subj_subj_data, cmap=cmap, vmin=min_corr, vmax=max_corr, edgecolors=edgecolors)
            # add correlations to mean
            # subjects_by_mean_ax = divider.append_axes("bottom", "7%", pad="60%")
            subjects_by_canonical_ax = divider.append_axes("bottom", "7%", pad="50%")

            # self.figure.add_axes(subjects_by_mean_ax)
            self.figure.add_axes(subjects_by_canonical_ax)
            # subjects_by_mean_ax.pcolor([self.subj_mean_data], cmap=cmap, vmin=min_corr, vmax=max_corr, edgecolors=edgecolors)
            subjects_by_canonical_ax.pcolor([self.subj_canonical_data], cmap=cmap, vmin=min_corr, vmax=max_corr, edgecolors=edgecolors)

            # subjects_by_mean_ax.set_ylabel(MEAN_Y_LABEL, rotation=0)
            # subjects_by_mean_ax.yaxis.set_label_coords(-0.5, 0)
            # subjects_by_mean_ax.set_xticks(np.arange(len(self.subj_mean_data)) + 0.5)
            # subjects_by_mean_ax.set_xticklabels(self.ids, fontsize=self.font_size, rotation=35)
            # subjects_by_mean_ax.set_yticks(np.arange(0))

            subjects_by_canonical_ax.set_ylabel(CANONICAL_Y_LABEL, rotation=0)
            subjects_by_canonical_ax.yaxis.set_label_coords(-0.5, 0)
            subjects_by_canonical_ax.set_xticks(np.arange(len(self.subj_canonical_data)) + 0.5)
            subjects_by_canonical_ax.set_xticklabels(self.ids, fontsize=self.font_size, rotation=35)
            subjects_by_canonical_ax.set_yticks(np.arange(0))

        else:
            heatmap_s_s = subjects_by_subjects_ax.pcolor(self.data, cmap=cmap, vmin=min_corr, vmax=max_corr, edgecolors=edgecolors)

        self.figure.colorbar(heatmap_s_s, cax=color_ax)
        self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', self.onClick)
        # plt.tight_layout()

    def plot_barchart(self):
        """
        Plots a barchart graph according to data
        """

        plt.gcf().subplots_adjust(bottom=0.2)

        self.figure.clear()
        ax = self.figure.gca()
        self.figure.suptitle(self.title)

        positive_indices = self.data[0]
        N = len(positive_indices)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars
        rects1 = ax.bar(ind, positive_indices, width, color='steelblue')
        negative_indices = self.data[1]
        rects2 = ax.bar(ind + width, negative_indices, width, color='mediumspringgreen')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('IOU')
        ax.set_xlabel('Subjects')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(self.ids, fontsize=self.font_size, rotation=35)

        ax.legend((rects1[0], rects2[0]), ('Positive', 'Negative'))
        def autolabel(rects, j):
            """
            Attach a text label above each bar displaying its height
            """
            for i in range(len(rects)):
                height = rects[i].get_height()
                index = self.data[j][i]
                ax.text(rects[i].get_x() + rects[i].get_width() / 2., 0.8 * height,
                        '{:01.2f}'.format(index),
                        ha='center', va='bottom', fontsize=self.font_size)
        autolabel(rects1, 0)
        autolabel(rects2, 1)

        # refresh canvas
        self.canvas.draw()



