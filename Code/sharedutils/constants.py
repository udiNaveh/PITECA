'''
A file with all the constants we need for the program.
I used anonymous objects to divide these constants into contexts (rightt now only
path constants and brain model constants, but you can ass your own too).
Not sure if this implementation is best practice for using constants.
If you find a better way make sure to refractor the code so that the modules
that use it still work.

'''

import numpy as np




MIN_TIME_UNITS = 300
NUM_FEATURES = 108
NUM_SPATIAL_FILTERS = 50

# Brain parcellations
STANDARD_BM = type('Dummy', (object,), {})
STANDARD_BM.N_LH = 29696
STANDARD_BM.N_RH = 29716
STANDARD_BM.N_CORTEX = STANDARD_BM.N_LH + STANDARD_BM.N_RH
STANDARD_BM.N_SC = 31870
STANDARD_BM.N_TOTAL_VERTICES = STANDARD_BM.N_LH + STANDARD_BM.N_RH + STANDARD_BM.N_SC
STANDARD_BM.CORTEX = np.array([i < STANDARD_BM.N_CORTEX for i in range(STANDARD_BM.N_TOTAL_VERTICES)])
STANDARD_BM.SUBCORTEX = np.bitwise_not(STANDARD_BM.CORTEX)


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
        return self.domain().name + "_" + self.name


class Domain(Enum):
    EMOTION = [Task(i) for i in range(1, 4)]
    GAMBLING = [Task(i) for i in range(4, 7)]
    LANGUAGE = [Task(i) for i in range(7, 10)]
    MOTOR = [Task(i) for i in range(10, 23)]
    RELATIONAL = [Task(i) for i in range(23, 26)]
    SOCIAL = [Task(i) for i in range(26, 29)]
    WM = [Task(i) for i in range(29, 48)]

AVAILABLE_TASKS = [Task.MATH_STORY ,
                       Task.TOM,
                       Task.MATCH_REL,
                       Task.TWO_BK,
                       Task.REWARD,
                       Task.FACES_SHAPES,
                       Task.T
                       ]

# Extensions
DTSERIES_EXT = ".dtseries.nii"
FEATS_EXT = "_features"
PREDICT_OUTPUT_EXT = "_predicted"

# Numbers
MAX_SUBJECTS = 25

# Dialogs
QUESTION_TITLE = "Question"
PROVIDE_INPUT_MSG = "Please provide input."
SELECT_ACTION_MSG = "Please select an action."
NAMING_CONVENTION_ERROR = "Some files don't apply naming convention. Please change names and try again."

# Existing features
EXIST_FEATS_MSG = "Some of the input scans seem to already have extracted features." \
                  " \nDo you want to use the existing features?"

UNEXPECTED_EXCEPTION_MSG = "An unexpected exception occurred"

# Quit prograss
ARE_YOU_SURE_MSG = "Are you sure you want to quit?"

