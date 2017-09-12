from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio

from sharedutils import dialog_utils
from GUI.analyze_working_thread import AnalysisTask


SUBJECTS_X_LABEL = "subjects_x"
SUBJECTS_Y_LABEL = "subjects_y"
MEAN_Y_LABEL = "with mean"
CANONICAL_Y_LABEL = "with canonical"


def is_heatmap_location(event):
    is_subjects_by_subjects = event.inaxes.get_xlabel() == SUBJECTS_X_LABEL and event.inaxes.get_ylabel() == SUBJECTS_Y_LABEL
    is_subjects_by_mean = event.inaxes.get_xlabel() == "" and event.inaxes.get_ylabel() == MEAN_Y_LABEL
    is_subjects_by_canonical = event.inaxes.get_xlabel() == "" and event.inaxes.get_ylabel() == CANONICAL_Y_LABEL
    return is_subjects_by_subjects or is_subjects_by_mean or is_subjects_by_canonical


class GraphicDlg(QDialog):
    def __init__(self, analysis_task, data, subjects, parent=None):
        super(GraphicDlg, self).__init__(parent)
        self.ids = [subject.subject_id for subject in subjects]
        self.data = data
        self.data = np.asarray([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.mock = np.asarray([[1, 2, 3]])

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

        if analysis_task == AnalysisTask.Analysis_Correlations:
            self.plot_heatmap()
        if analysis_task == AnalysisTask.Compare_Correlations:
            self.plot_heatmap()


    def onClick(self, event):
        if event.button == 1 and event.xdata and event.ydata:
            if is_heatmap_location(event):
                subject_x_index = math.floor(event.xdata)
                subject_y_index = math.floor(event.ydata)
                correlation = self.data[subject_x_index, subject_y_index]
                self.correlation_label.setText("Value: {:.2f}".format(correlation))

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
            raise Exception()

    def plot_heatmap(self):

        # a label to show correlation
        self.correlation_label = QtWidgets.QLabel('Click on entry to see the exact correlation value')
        self.layout.addWidget(self.correlation_label)

        self.figure.clear()

        # create an axis
        subjects_by_subjects_ax = self.figure.gca()
        subjects_by_subjects_ax.set_xlabel(SUBJECTS_X_LABEL)
        subjects_by_subjects_ax.set_ylabel(SUBJECTS_Y_LABEL)
        subjects_by_subjects_ax.set_xticks(np.arange(6) + 0.5)
        subjects_by_subjects_ax.set_xticklabels(self.ids)
        subjects_by_subjects_ax.set_yticks(np.arange(6) + 0.5)
        subjects_by_subjects_ax.set_yticklabels(self.ids)

        # plot data
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(subjects_by_subjects_ax)
        color_ax = divider.append_axes("right", "3%", pad="1%")
        subjects_by_mean_ax = divider.append_axes("bottom", "7%", pad="30%")
        subjects_by_canonical_ax = divider.append_axes("bottom", "7%", pad="25%")
        heatmap_s_s = subjects_by_subjects_ax.pcolor(self.data)
        self.figure.colorbar(heatmap_s_s, cax=color_ax)
        self.figure.add_axes(subjects_by_mean_ax)
        self.figure.add_axes(subjects_by_canonical_ax)
        subjects_by_mean_ax.pcolor(self.mock)
        subjects_by_canonical_ax.pcolor(self.mock)

        subjects_by_mean_ax.set_ylabel(MEAN_Y_LABEL, rotation=0)
        subjects_by_mean_ax.yaxis.set_label_coords(-0.08, +1)
        subjects_by_mean_ax.set_xticks(np.arange(3) + 0.5)
        subjects_by_mean_ax.set_xticklabels(self.ids)
        subjects_by_mean_ax.set_yticks(np.arange(0))

        subjects_by_canonical_ax.set_ylabel(CANONICAL_Y_LABEL, rotation=0)
        subjects_by_canonical_ax.yaxis.set_label_coords(-0.05, +1)
        subjects_by_canonical_ax.set_xticks(np.arange(3) + 0.5)
        subjects_by_canonical_ax.set_xticklabels([1, 2, 3])
        subjects_by_canonical_ax.set_yticks(np.arange(0))

        self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', self.onClick)


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



