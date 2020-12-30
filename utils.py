import zipfile
import os.path


def unzip(filepath, directory_to_extract_to):
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def unzip_files(path, extract_to, files_to_unzip, keep_zip):
    
    
        for filename in files_to_unzip:
            try:
                # .zip file exists and .csv doesn't
                if not os.path.isfile(extract_to + os.path.splitext(filename)[0]) and os.path.isfile(path + filename):
                    unzip(path + filename , extract_to)
                    print(filename, "unzipped")
                # .zip and .csv files don't file exist
                elif not os.path.isfile(extract_to + os.path.splitext(filename)[0]) and not os.path.isfile(path + filename):
                    print("{} Not Found".format(filename))
                
                if not keep_zip:
                    os.remove(path + filename)
            except FileNotFoundError:
                print("Wrong file or file path")
                continue

