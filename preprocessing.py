import utils
from os import path, listdir, walk
import sys


def extract_files(filepath, extract_path, files_to_unzip, keep_zip):
    '''
    Unzips the files
    '''
    utils.unzip_files(filepath, extract_path, files_to_unzip, keep_zip)

def list_files(directory, extension):
    '''
    input: directory path, file extension
    ouput: list of files in the directory with the file extension
    '''
    for (dirpath, dirnames, filenames) in walk(directory):
        return [f for f in filenames if f.endswith(extension)]

if __name__ == "__main__":
    '''
    input: filepath, keep_zip, extract_all, files_to_unzip (if not extract_all), extract_here, extract_path (if not extract_here)
    '''
    filepath = sys.argv[1]
    keep_zip = int(sys.argv[2])
    x = 3
    extract_all = int(sys.argv[3])
    if not extract_all:
        files_to_unzip = sys.argv[x+1].strip('[]').split(',')
        x += 1
    else:
        files_to_unzip = list_files(filepath, '.zip')
    extract_here = int(sys.argv[x+1])
    extract_path = filepath
    if not extract_here:
        x += 1
        extract_path = sys.argv[x+1]
    extract_files(filepath, extract_path, files_to_unzip, keep_zip)