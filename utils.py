import zipfile
import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
wordnet_lemmatizer = WordNetLemmatizer()


def unzip(filepath, extract_to):
    """
    Unzips a file at filepath and extracts all the contents to the extract_to directory
    filepath: path to the file to unzip
    extract_to: directory/path to unzip the files at
    """
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def list_files(directory=".", extension=""):
    """
    Returns list of all files ending with extension in directory
    input: directory path, file extension
    output: list of files in the directory with the file extension
    """
    for (_, _, filenames) in os.walk(directory):
        return [f for f in filenames if f.endswith(extension)]


def unzip_files(path=".", extract_to=None, files_to_unzip=[], keep_zips=False):
    """
    Unzips all files_to_unzip at path to extract_to directory
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
    Returns text after removing html formatting tags
    html_text: text along with HTML formatting tags
    """
    return BeautifulSoup(html_text, "html.parser").get_text()


def remove_stopwords_without_tokenize(text):
    """
    Returns text after removing stopwords without tokenization
    text: string from which stopwords are to be removed
    """
    return (" ").join(
        [word for word in text.split(" ") if not word in stopwords.words("english")]
    )


def remove_stopwords_with_tokenize(text):
    """
    Returns text after removing stopwords with tokenization
    text: string from which stopwords are to be removed
    """
    return (" ").join(
        [word for word in word_tokenize(text) if not word in stopwords.words("english")]
    )


def get_wordnet_pos(word):
    """
    Maps POS tag to the first character lemmatize() accepts
    word: word for which POS tag is to be assigned
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(text):
    """
    Returns lemmatized text
    text: string to be lemmatized
    """
    punctuations = "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"
    return (" ").join(
        [
            wordnet_lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in word_tokenize(text)
            if w not in punctuations
        ]
    )
