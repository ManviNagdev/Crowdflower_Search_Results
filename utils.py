import os
from zipfile import ZipFile

from bs4 import BeautifulSoup
import fasttext
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import download as nltk_download
from nltk import pos_tag as nltk_pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    return BeautifulSoup(html_text, "lxml").text


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
    pd.DataFrame(train_bow.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/train_bow.tsv", sep="\t"
    )

    test_bow = vectorizer.transform(test)
    pd.DataFrame(test_bow.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/test_bow.tsv", sep="\t"
    )


def tf_idf(train, test=None):
    """
    Converts train file into matrix of TF-IDF features and transforms test file into document-term matrix
    train: List of training sentences to be used to form vocabulary and then transformed
    test: List of testing sentences to be tranformed
    """
    vectorizer = TfidfVectorizer()
    train_tfidf = vectorizer.fit_transform(train)
    pd.DataFrame(train_tfidf.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/train_tfidf.tsv", sep="\t"
    )

    test_tfidf = vectorizer.transform(test)
    pd.DataFrame(test_tfidf.toarray(), columns=vectorizer.get_feature_names()).to_csv(
        "preprocessed_data/test_tfidf.tsv", sep="\t"
    )


def word_vector(train, test=None, features=[]):
    """
    Converts words to corresponding vectors using gensim's Word2Vec model
    train: data used to train Word2Vec model
    test: testing data to be tranformed
    features: list of columns to be vectorized
    """
    sentence_list = []
    for feature in features:
        for item in train[feature]:
            sentence_list.append(list(item.split(" ")))

    word2vec_model = Word2Vec(sentence_list, min_count=1, size=300, workers=2)

    # saving the model
    word2vec_model.save("word2vec.model")
    word2vec_model.save("word2vec.bin")
    df_train = [train]
    df_test = [test]
    for feature in features:
        train[feature] = train[feature].str.split()
        test[feature] = test[feature].str.split()
        train_embeddings = get_word2vec_embeddings(word2vec_model, train[feature])
        df_train.append(pd.DataFrame(np.asarray(train_embeddings)))
        test_embeddings = get_word2vec_embeddings(word2vec_model, test[feature])
        df_test.append(pd.DataFrame(np.asarray(test_embeddings)))

    train_final = pd.concat(df_train, axis=1)
    test_final = pd.concat(df_test, axis=1)

    train_final.to_csv("preprocessed_data/train_w2v.tsv", sep="\t")
    test_final.to_csv("preprocessed_data/test_w2v.tsv", sep="\t")


def glove_embeddings(train, test=None, features=[]):
    """
    Converts words to corresponding vectors using GloVe pre-trained word embeddings
    train: training data to be tranformed
    test: testing data to be tranformed
    features: list of columns to be vectorized
    """
    GLOVE_DIR = "D:\Machine Learning\glove.6B"
    glove_filename = "glove.6B.300d.txt"
    glove_file_path = os.path.join(GLOVE_DIR, glove_filename)
    glove_output_file = "glove_embeddings.txt"
    glove2word2vec(glove_file_path, glove_output_file)

    # load the Stanford GloVe model
    glove_model = KeyedVectors.load_word2vec_format(glove_output_file, binary=False)

    df_train = [train]
    df_test = [test]

    for feature in features:
        train[feature] = train[feature].str.split()
        test[feature] = test[feature].str.split()
        train_embeddings = get_word2vec_embeddings(glove_model, train[feature])
        df_train.append(pd.DataFrame(np.asarray(train_embeddings)))
        test_embeddings = get_word2vec_embeddings(glove_model, test[feature])
        df_test.append(pd.DataFrame(np.asarray(test_embeddings)))

    train_final = pd.concat(df_train, axis=1)
    test_final = pd.concat(df_test, axis=1)

    train_final.to_csv("preprocessed_data/train_glove.tsv", sep="\t")
    test_final.to_csv("preprocessed_data/test_glove.tsv", sep="\t")


def fasttext_embeddings(train, test=None, features=[]):
    """
    Converts words to corresponding vectors using fasttext word embeddings
    train: data used to train fasttext model
    test: testing data to be tranformed
    features: list of columns to be vectorized
    """

    sentence_list = []
    for feature in features:
        for item in train[feature]:
            sentence_list.append(list(item.split(" ")))

    # create input file for training fasttext model
    pd.DataFrame(sentence_list).to_csv("preprocessed_data/fasttext_input.tsv", sep="\t")
    fasttext_model = fasttext.train_unsupervised(
        "preprocessed_data/fasttext_input.tsv", model="skipgram", minCount=1, dim=300
    )

    # saving the model
    fasttext_model.save_model("fasttext_model.bin")
    df_train = [train]
    df_test = [test]
    for feature in features:
        train[feature] = train[feature].str.split()
        test[feature] = test[feature].str.split()
        train_embeddings = get_word2vec_embeddings(fasttext_model, train[feature])
        df_train.append(pd.DataFrame(np.asarray(train_embeddings)))
        test_embeddings = get_word2vec_embeddings(fasttext_model, test[feature])
        df_test.append(pd.DataFrame(np.asarray(test_embeddings)))

    train_final = pd.concat(df_train, axis=1)
    test_final = pd.concat(df_test, axis=1)

    train_final.to_csv("preprocessed_data/train_fasttext.tsv", sep="\t")
    test_final.to_csv("preprocessed_data/test_fasttext.tsv", sep="\t")


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    """
    Returns the average vector for the words in the token_list
    token_list: list of words
    vector: model to be used for converting word to vec
    generate_missing: bool for generating the average vec for words not present in the model vocabulary
    k: vector dimension
    """
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [
            vector[word] if word in vector else np.random.rand(k)
            for word in tokens_list
        ]
    else:
        vectorized = [
            vector[word] if word in vector else np.zeros(k) for word in tokens_list
        ]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, df, generate_missing=False):
    """
    Returns a list of word to vec embeddings with average value for the vectors
    vectors: model to be used for converting word to vec
    df: pd.DataFrame on which the embedding is to be applied
    generate_missing: bool for generating the average vec for words not present in the model vocabulary
    """
    embeddings = df.apply(
        lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing)
    )
    return list(embeddings)


def training_testing_split(filepath, label, drop_features=[]):
    """
    Performs the train_test_split on the tsv file specified
    filepath: file location
    label: column name which specifies the label to be predicted
    drop_features: columns not to be considered while training
    """
    train = pd.read_csv(filepath, sep="\t")
    filename = os.path.splitext(os.path.basename(filepath))[0]

    y = train[label]
    train = train.drop(drop_features, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        train, y, test_size=0.33, random_state=42
    )

    train_final = pd.concat(
        [
            pd.DataFrame(np.asarray(X_train)),
            pd.DataFrame({"label": np.asarray(y_train)}),
        ],
        axis=1,
    )
    train_final.to_csv("preprocessed_data/train_split_" + filename + ".tsv", sep="\t")

    test_final = pd.concat(
        [
            pd.DataFrame(np.asarray(X_test)),
            pd.DataFrame({"label": np.asarray(y_test)}),
        ],
        axis=1,
    )
    test_final.to_csv("preprocessed_data/test_split_" + filename + ".tsv", sep="\t")


def model_training(X_train, X_test, y_train, y_test, model):
    """
    returns MCC score after training the specified model
    X_train: dataframe for training data
    y_train: class label for the X_train
    X_test: dataframe for testing data
    y_test: class label for the X_test
    model: model that is to be trained
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return matthews_corrcoef(y_test, y_pred)