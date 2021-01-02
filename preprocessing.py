import utils
from os import path, listdir, walk
import sys


if __name__ == "__main__":
    """
    input: filepath, extract_path (if not extract_here)
    """
    filepath = sys.argv[1]
    try:
        extract_path = sys.argv[2]
    except:
        extract_path = None
    utils.unzip_files(path=filepath, extract_to=extract_path, files_to_unzip=utils.list_files(directory=filepath, extension=".zip"))