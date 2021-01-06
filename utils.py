import os
from zipfile import ZipFile

from bs4 import BeautifulSoup
from nltk import download as nltk_download
from nltk import pos_tag as nltk_pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk_download("punkt")
nltk_download("wordnet")
nltk_download("stopwords")
nltk_download("averaged_perceptron_tagger")
wordnet_lemmatizer = WordNetLemmatizer()


def unzip(filepath, extract_to):
    """
    Unzips a file at filepath and extracts all the contents to the extract_to directory
    filepath: path to the file to unzip
    extract_to: directory/path to unzip the files at
    """
    with ZipFile(filepath, "r") as zip_ref:
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
    return BeautifulSoup(html_text, "html.parser", store_line_numbers=False).get_text()


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
    tag = nltk_pos_tag([word])[0][1][0].upper()
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


def bag_of_words(train, test=None):
    """
    Converts train file into matrix of token counts and transforms test file into document-term matrix
    train: List of training sentences to be used to form vocabulary and then transformed
    test: List of testing sentences to be tranformed
    """
    vectorizer = CountVectorizer()
    train_bow = vectorizer.fit_transform(train)
    DataFrame(train_bow.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/train_bow.csv"
    )

    test_bow = vectorizer.transform(test)
    DataFrame(test_bow.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/test_bow.csv"
    )


def tf_idf(train, test=None):
    """
    Converts train file into matrix of TF-IDF features and transforms test file into document-term matrix
    train: List of training sentences to be used to form vocabulary and then transformed
    test: List of testing sentences to be tranformed
    """
    vectorizer = TfidfVectorizer()
    train_tfidf = vectorizer.fit_transform(train)
    DataFrame(train_tfidf.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/train_tfidf.csv"
    )

    test_tfidf = vectorizer.transform(test)
    DataFrame(test_tfidf.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/test_tfidf.csv"
    )
