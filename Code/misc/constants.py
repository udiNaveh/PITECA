from enum import Enum


class Contrasts(Enum):
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


class Tasks(Enum):
    EMOTION = [Contrasts(i) for i in range(1, 4)]
    GAMBLING = [Contrasts(i) for i in range(4, 7)]
    LANGUAGE = [Contrasts(i) for i in range(7, 10)]
    MOTOR = [Contrasts(i) for i in range(10, 23)]
    RELATIONAL = [Contrasts(i) for i in range(23, 26)]
    SOCIAL = [Contrasts(i) for i in range(26, 29)]
    WM = [Contrasts(i) for i in range(29, 48)]

'''
Strings
'''
DTSERIES_EXT = ".dtseries.nii"
FEATS_EXT = "_features.py"

"""
Hard coded values
"""
CONFIG_PATH = "C:/Users/User/PycharmProjects/Sadna/PITECA/GUI/config.txt"