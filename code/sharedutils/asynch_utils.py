'''
the name doesn't suggest much - need to find another name for this module.
the idea is to put here helper methods that are called from within processes 
in the controller, trigger something in the UI (for example showing a message/dialog box). 

'''



def do_use_existing_features(feature_path):
    '''
    trigger a UI event - ask the user whether to use an existing features file, to replace it, or 
    to extract new features an keep both files.
    :param feature_path: 
    :return: 
    '''
    # just for now
    answer = input("the features for subject $$$ are in file {}. use existing?".format(feature_path))

    return answer, feature_path


def do_convert_to_CIFTI2(cifti_path):
    '''
    ask the user if to convert a file to cifti 2 format if it was not in this format.
    :param cifti_path: 
    :return: bool
    '''
    answer = False
    cifti2_path = ''
    ####
    #do something
    #####
    return answer, cifti2_path
