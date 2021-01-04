import zipfile
import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
wordnet_lemmatizer = WordNetLemmatizer()


def unzip(filepath, extract_to):
    """
    filepath: path to the file to unzip
    extract_to: directory/path to unzip the files at
    """
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def list_files(directory=".", extension=""):
    """
    input: directory path, file extension
    output: list of files in the directory with the file extension
    """
    for (_, _, filenames) in os.walk(directory):
        return [f for f in filenames if f.endswith(extension)]


def unzip_files(path=".", extract_to=None, files_to_unzip=[], keep_zips=False):
    """
    path: path to zip files
    extract_to: directory/path to unzip the files at (if None, extracts to `path`)
    files_to_unzip: list of files to unzip
    keep_zips: whether to keep the .zip files
    """
    path = os.path.abspath(path)
    if not extract_to:
        extract_to = path
    else:
        extract_to = os.path.abspath(extract_to)
    for filename in files_to_unzip:
        try:
            if not os.path.isfile(
                os.path.join(extract_to, os.path.splitext(filename)[0])
            ):  # unzipped files do not exist
                if os.path.isfile(os.path.join(path, filename)):  # .zip files exist
                    unzip(os.path.join(path, filename), extract_to)
                else:  # .zip files do not exist
                    print("{} not found".format(filename))

            if not keep_zips:
                os.remove(os.path.join(path, filename))

        except FileNotFoundError:
            print("Wrong file or file path: {}".format(filename))
            continue


def html2text(html_text):
    """
    html_text: text along with HTML formatting tags
    """
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text()


def remove_stopwords_without_tokenize(text):
    """
    text: string from which stopwords are to be removed
    """
    text_without_sw = [
        word for word in text.split(" ") if not word in stopwords.words("english")
    ]
    return (" ").join(text_without_sw)


def remove_stopwords_with_tokenize(text):
    """
    text: string from which stopwords are to be removed
    """
    text_tokens = word_tokenize(text)
    text_without_sw = [
        word for word in text_tokens if not word in stopwords.words("english")
    ]
    return (" ").join(text_without_sw)


def lemmatize(text):
    """
    text: string to be lemmatized
    """
    punctuations = "?:!.,;"
    text_tokens = word_tokenize(text)
    lemmatized_text = [
        wordnet_lemmatizer.lemmatize(word, pos="v")
        for word in text_tokens
        if not word in punctuations
    ]
    return (" ").join(lemmatized_text)