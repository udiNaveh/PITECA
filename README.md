# PITECA

PITECA project is a standalone desktop application for predicting individual task-evoked cortical activation based on restring-state fMRI data.

### Dependencies

The following should be installed prior to running the project:

- Workbench (https://www.humanconnectome.org/): After installation 'wb_view' should be added to path
- Python 3.5 (https://www.python.org/)
- PyQt 5.6 (https://www.riverbankcomputing.com/software/pyqt/download5)
- TensorFlow 1.1 (https://www.tensorflow.org/)
- Numpy 1.12 (http://www.numpy.org/)
- cifti 1.0 (https://github.com/MichielCottaar/cifti)

### Usage

To open the app, just run 'GUI/main_controller.py'. 

Input resting-state files should apply the following conventions:
1. Should be a dense time series CIFTI2 file (.dtseries.nii)
2. At least 200 time-points scans with TR < 3.5s (RECOMMENDED)
3. Filename begins with "SubjectID_" where "SubjectID" is an integer (e.g,. "12345_restingscan.dtseries.nii", but not "restingscan_12345.dtseries.nii").
4. Number of input files is limited to 25 per run. 

Each output file (henceforth, prediction map) describes a subject's predicted cortical activation during a task. 
The files are saved as dense scalar CIFTI2 files (dscalar.nii) where the value at each brainordinate represents the COPEs for the activity value in that brainordinate for the task. 

At the end of a prediction process, the specified output folder set in the settings will have the following hierarchy:
1. Subfolder for each domain selected
2. Subfolder for each contrast selected in the domain
3. Predicted activation files during contrast performance for each subject

Resulted files are saved under the following name-format: "SubjectID_Domain_Contrast.dscalar.nii".