"""
This module contains wrapper functions for command line calls
"""

import os
import subprocess
import definitions


def system_call(command):
    p = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    return p


def run_wb_command(args):
    return system_call(['wb_command'] + args)

def run_wb_view(args):
    subprocess.Popen(['wb_view'] + args)

def run_command(command, args):
    assert isinstance(command, str)
    assert isinstance(args, list)
    line = "{0} {1}".format(command, str.join(args, ' '))
    system_call(line)


def convert_to_CIFTI2(input_cifti, output_cifti2):
    return run_wb_command(["-file-convert", "-cifti-version-convert", input_cifti, '2', output_cifti2])

def show_maps_in_wb_view(cifti_paths):
    if isinstance(cifti_paths, str):
        cifti_paths = [cifti_paths]
    run_wb_view([definitions.DEFAULT_LEFT_SURFACE_PATH, definitions.DEFAULT_RIGHT_SURFACE_PATH] + cifti_paths)

