import numpy as np
import os.path as path

'''
A file with all the constants we need for the program.
I used anonymous objects to divide these constants into contexts (rightt now only
path constants and brain model constants, but you can ass your own too).
Not sure if this implementation is best practice for using constants.
If you find a better way make sure to refractor the code so that the modules
that use it still work.

'''



MIN_TIME_UNITS = 300
NUM_FEATURES = 108

# Brain parcellations
STANDART_BM = type('Dummy', (object,), {})
STANDART_BM.N_LH = 29696
STANDART_BM.N_RH = 29716
STANDART_BM.N_CORTEX = STANDART_BM.N_LH + STANDART_BM.N_RH
STANDART_BM.N_SC = 31870
STANDART_BM.N_TOTAL_VERTICES = STANDART_BM.N_LH + STANDART_BM.N_RH + STANDART_BM.N_SC
STANDART_BM.CORTEX = np.array([i < STANDART_BM.N_CORTEX for i in range(STANDART_BM.N_TOTAL_VERTICES)])
STANDART_BM.SUBCORTEX = np.bitwise_not(STANDART_BM.CORTEX)

# absolute paths to directories or files
PATHS = type('Dummy', (object,), {})
PATHS.PITECA_DIR = path.join(path.expanduser('~ASUS'), 'Dropbox', 'PITECA') # change with your own
PATHS.DATA_DIR = path.join(PATHS.PITECA_DIR, 'Data')
PATHS.GARBAGE = path.join(PATHS.DATA_DIR, 'garbage')
PATHS.MATLAB_CODE_DIR = path.join(PATHS.PITECA_DIR, 'Code', 'MATLAB')
PATHS.WB_TUTORIAL_DATA = r"D:\Projects\HCP_WB_Tutorial_1.0\HCP_WB_Tutorial_1.0" # change with your own
PATHS.SC_CLUSTERS = path.join(PATHS.DATA_DIR, 'SC_clusters.dtseries.nii')
PATHS.ICA_LR_MATCHED = path.join(PATHS.DATA_DIR, 'ica_LR_MATCHED.dtseries.nii')
PATHS.EXTRACTED_FEATURES_PATH = r"D:\Projects\PITECA\Data\extracted features"


from enum import Enum

class Task(Enum):
    FACES = 1
    SHAPES = 2
    FACES_SHAPES = 3
    PUNISH = 4
    REWARD = 5
    PUNISH_REWARD = 6
    MATH = 7
    STORY = 8
    MATH_STORY = 9
    CUE = 10
    LF = 11
    LH = 12
    RF = 13
    RH = 14
    T = 15
    AVG = 16
    CUE_AVG = 17
    LF_AVG = 18
    LH_AVG = 19
    RF_AVG = 20
    RH_AVG = 21
    T_AVG = 22
    MATCH = 23
    REL = 24
    MATCH_REL = 25
    RANDOM = 26
    TOM = 27
    RANDOM_TOM = 28
    TWO_BK_BODY = 29
    TWO_BK_FACE = 30
    TWO_BK_PLACE = 31
    TWO_BK_TOOL = 32
    O_BK_BODY = 33
    O_BK_FACE = 34
    O_BK_PLACE = 35
    O_BK_TOOL = 36
    TWO_BK = 37
    O_BK = 38
    TWO_BK_O_BK = 39
    BODY = 40
    FACE = 41
    PLACE = 42
    TOOL = 43
    BODY_AVG = 44
    FACE_AVG = 45
    PLACE_AVG = 46
    TOOL_AVG = 47

    def domain(self):
        for d in Domain:
            if self in d.value:
                break
        return d

    @property
    def full_name(self):
        return self.domain.name + "_" + self.name


class Domain(Enum):
    EMOTION = [Task(i) for i in range(1, 4)]
    GAMBLING = [Task(i) for i in range(4, 7)]
    LANGUAGE = [Task(i) for i in range(7, 10)]
    MOTOR = [Task(i) for i in range(10, 23)]
    RELATIONAL = [Task(i) for i in range(23, 26)]
    SOCIAL = [Task(i) for i in range(26, 29)]
    WM = [Task(i) for i in range(29, 48)]

'''
Extensions
'''
DTSERIES_EXT = ".dtseries.nii"
FEATS_EXT = "_features"
PREDICT_OUTPUT_EXT = "_predicted"

'''
Dialogs
'''
QUESTION_TITLE = "PITECA - Dialog"

# Existing features
EXIST_FEATS_MSG = "Some of the input scans seem to already have extracted features." \
                  " \n Do you want to use the existing features?"

"""
Hard coded values
"""
TMP_FEATURES_PATH = "C:/Users/User/Sadna/src/git/PITECA/Code/misc/Mock Files/features"
# TODO: use the path in the configurations instead
