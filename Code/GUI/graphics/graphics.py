from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio

from sharedutils import dialog_utils


X_LABEL = "subjects_x"
Y_LABEL = "subjects_y"


def is_heatmap_location(event):
    return event.inaxes.get_xlabel() == X_LABEL and event.inaxes.get_ylabel() == Y_LABEL


class GraphicDlg(QDialog):
    def __init__(self, analysis_task, data, parent=None):
        super(GraphicDlg, self).__init__(parent)
        self.data = np.asarray([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])

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
        self.save_button.clicked.connect(self.save_data)

        # a label to show correlation
        self.correlation_label = QtWidgets.QLabel('Click on entry to see the exact correlation value')

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.correlation_label)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.plot_barchart()


    def onClick(self, event):

        if event.button == 1 and event.xdata and event.ydata:
            if is_heatmap_location(event):
                subject_x_index = math.floor(event.xdata)
                subject_y_index = math.floor(event.ydata)
                correlation = self.data[subject_x_index, subject_y_index]
                self.correlation_label.setText(
                    "The correlation between subject {:.0f} and {:.0f} is {:.2f}"
                        .format(subject_x_index, subject_y_index, correlation))

    def save_data(self):
        name, extension = dialog_utils.save_file('*.mat ;; *.numpy')
        extension = extension.split('.')[1]

        if extension == 'mat':
            sio.savemat(name, {'data': self.data})
        elif extension == 'numpy':
            self.data.tofile(name)
        else:
            # @error_handling
            print("oh oh")

    def plot_heatmap(self):

        self.figure.clear()

        # create an axis
        ax = self.figure.gca()
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_xticks(np.arange(6) + 0.5)
        ax.set_xticklabels([1, 2, 3])
        ax.set_yticks(np.arange(6) + 0.5)
        ax.set_yticklabels([1, 2, 3])

        # plot data
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "3%", pad="1%")
        heatmap = ax.pcolor(self.data)
        self.figure.colorbar(heatmap, cax=cax)

        # refresh canvas
        self.canvas.draw()

        self.canvas.mpl_connect('button_press_event', self.onClick)


    def plot_barchart(self):

        self.figure.clear()
        ax = self.figure.gca()

        men_means = (20, 35, 30, 35, 27)
        men_std = (2, 3, 4, 1, 2)
        N = 5

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        rects1 = ax.bar(ind, men_means, width, color='steelblue', yerr=men_std)

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



