import getopt
import sys

from pandas import read_csv
from tqdm import tqdm

import utils

if __name__ == "__main__":
    """
    input: --lemmatize=, filepath, extract_path (if not extract_here)
    """
    options, remainder = getopt.getopt(sys.argv[1:], "o:v", ["lemmatize="])
    lemmatize = False
    for opt, arg in options:
        if opt in ("-l", "--lemmatize"):
            lemmatize = arg
    filepath = remainder[0]

    try:
        extract_path = remainder[1]
    except:
        extract_path = None

    utils.unzip_files(
        path=filepath,
        extract_to=extract_path,
        files_to_unzip=utils.list_files(directory=filepath, extension=".zip"),
    )

    # load dataset
    train = read_csv("data/train.csv")
    test = read_csv("data/test.csv")

    # drop NaN values
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    # convert to lowercase
    train["product_description"] = train["product_description"].str.lower()
    train["product_title"] = train["product_title"].str.lower()
    train["query"] = train["query"].str.lower()

    test["product_description"] = test["product_description"].str.lower()
    test["product_title"] = test["product_title"].str.lower()
    test["query"] = test["query"].str.lower()

    # convert html text to normal text
    train["product_description"] = train["product_description"].apply(utils.html2text)
    test["product_description"] = test["product_description"].apply(utils.html2text)

    # remove stopwords
    for i, row in tqdm(train.iterrows()):
        train.loc[i, "product_description"] = utils.remove_stopwords_without_tokenize(
            row["product_description"]
        )
    train.to_csv("preprocessed_data/train_removed_stopwords.csv", sep="\t")

    for i, row in tqdm(test.iterrows()):
        test.loc[i, "product_description"] = utils.remove_stopwords_without_tokenize(
            row["product_description"]
        )
    test.to_csv("preprocessed_data/test_removed_stopwords.csv", sep="\t")
    print("Stopwords Removal done")

    # lemmatize
    if lemmatize == "True":
        for i, row in tqdm(train.iterrows()):
            train.loc[i, "product_description"] = utils.lemmatize(
                row["product_description"]
            )
        train.to_csv("preprocessed_data/train_lemmatized.csv", sep="\t")

        for i, row in tqdm(test.iterrows()):
            test.loc[i, "product_description"] = utils.lemmatize(
                row["product_description"]
            )
        test.to_csv("preprocessed_data/test_lemmatized.csv", sep="\t")
    print("Lemmatization done")

    # drop NaN values
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    # removing extra spaces
    train["product_description"] = train.str.replace(r"[^\w\s]", "")
    test["product_description"] = test.str.replace(r"[^\w\s]", "")

    # removing digits
    train["product_description"] = train.apply(
        lambda x: " ".join(x for x in x.split() if not x.isdigit())
    )
    test["product_description"] = test.apply(
        lambda x: " ".join(x for x in x.split() if not x.isdigit())
    )

    # create bag of words
    utils.bag_of_words(train["product_description"], test=test["product_description"])
    print("BOW done")

    # create tfidf
    utils.tf_idf(train["product_description"], test=test["product_description"])
    print("TFIDF done")

    # create word2vec
    utils.word_vector(train["product_description"], test["product_description"])
    print("Word2Vec done")

    # create vec using Glove embeddings
    utils.glove_embeddings(train["product_description"], test["product_description"])
    print("Glove done")

    # create vec using fasttext
    utils.fasttext_embeddings(train["product_description"], test["product_description"])
    print("Fasttext done")