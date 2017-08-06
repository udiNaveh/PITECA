
import os
import subprocess
import sharedutils.constants
import cifti


'''
wrapper functions for command line calls, especially wb_command but not necessarily.
didn't do much here. Still need to figure out what to do with the return values:
what happens when the command returned an error mesage, etc.
'''


def system_call(command):
    p = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    return p


def run_wb_command(args):
    return system_call(['wb_command'] + args)


def run_command(command, args):
    assert isinstance(command, str)
    assert isinstance(args, list)
    line = "{0} {1}".format(command, str.join(args, ' '))
    system_call(line)


########
# useful workbench commands to be added here
########


def convert_to_CIFTI2(input_cifti, output_cifti2):
    return run_wb_command(["-file-convert", "-cifti-version-convert", input_cifti, '2', output_cifti2])
