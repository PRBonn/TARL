import subprocess
import os
from pathlib import Path
import fileinput
import sys


def replaceAll(file: str, searchExp: str, replaceExp: str, replace_line=False):
    """Replaces all strings which match searchExp with the replaceExp

    Args:
        file ([str]): The file you want to change
        searchExp ([str]): The text you want to search
        replaceExp ([str]): The text you want it to replace
    """
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            if replace_line:
                line = replaceExp
            else:
                line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)


# Insert Package name and rename folder
old_name = 'tarl'
if os.path.isdir(old_name):
    # old_name = input('Please insert the old package name you want to rename: ')
    pkg_name = input('Please insert package name: ')
    os.rename(old_name, pkg_name)

    # rename all the package imports
    for path in Path(pkg_name).rglob('*.py'):
        replaceAll(path, f'import {old_name}', f'import {pkg_name}')
        pass

    replaceAll('setup.py', "pkg_name = ",
               f"pkg_name = '{pkg_name}'\n", replace_line=True)

file_dir = os.path.dirname(os.path.realpath(__file__))
command = f'pip3 install -U -e {file_dir}'
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
