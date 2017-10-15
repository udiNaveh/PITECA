# PITECA

PITECA project is a standalone desktop application for predicting individual task-evoked cortical activation based on restring-state fMRI data.
It was done as part of a workshop in computational methods in neuroscience, Tel Aviv University, Summer 2017.

### Dependencies

The following should be installed prior to running the project:

- Workbench (https://www.humanconnectome.org/): After installation 'wb_view' should be added to path
- Python 3.5 + (https://www.python.org/)
- PyQt 5.6 (https://www.riverbankcomputing.com/software/pyqt/download5)
- TensorFlow 1.1 + (https://www.tensorflow.org/)
- cifti 1.0 (https://github.com/MichielCottaar/cifti)

### Use

To open the app gui, just run 'Code/GUI/main_controller.py'.

Input resting-state files should apply the following conventions:
1. Should be a dense time series CIFTI file (.dtseries.nii)
2. At least 300 time-points scans with TR < 3.5s
3. Filename begins with "SubjectID_" where "SubjectID" is an integer (e.g,. "12345_restingscan.dtseries.nii",
    but not "restingscan_12345.dtseries.nii").
4. Number of input files is limited to 25 per run. 

Each output file (henceforth, prediction map) describes a subject's predicted amount of activation related to a specific task contrast. Predictions are available for task contrast from 7 different domains from teh HCP fmri data.
The files are saved as dtseries.nii where the value at each brainordinate represents the activity value in that brainordinate for the task.
The prediction model implementation was based on the work of Tavor et al 2016 Science paper.

At the end of a prediction process, the specified output folder set in the settings will have the following hierarchy:
1. Subfolder for each domain selected
2. Subfolder for each contrast selected in the domain
3. Predicted activation files during contrast performance for each subject

Resulted files are saved under the following name-format: "SubjectID_Domain_Contrast.dtseries.nii".

In addition, several basic analysis tools for interpreting the results are available in the analysis tab.



### Source code:
The code for using and learning the prediction model is found in Code/model.
The are currently 5 different prediction models available (the pne used by the GUI application can be chosen in the 'settings' tab).
Other models can be added by implementing the IModel abstract base class defined in Code/mode/models.py.
