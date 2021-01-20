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
        train.loc[i, "query"] = utils.remove_stopwords_without_tokenize(row["query"])
    train.to_csv("preprocessed_data/train_removed_stopwords.tsv", sep="\t")

    for i, row in tqdm(test.iterrows()):
        test.loc[i, "product_description"] = utils.remove_stopwords_without_tokenize(
            row["product_description"]
        )
        test.loc[i, "query"] = utils.remove_stopwords_without_tokenize(row["query"])
    test.to_csv("preprocessed_data/test_removed_stopwords.tsv", sep="\t")
    print("Stopwords Removal done")

    # lemmatize
    if lemmatize == "True":
        for i, row in tqdm(train.iterrows()):
            train.loc[i, "product_description"] = utils.lemmatize(
                row["product_description"]
            )
            train.loc[i, "query"] = utils.lemmatize(row["query"])
        train.to_csv("preprocessed_data/train_lemmatized.tsv", sep="\t")

        for i, row in tqdm(test.iterrows()):
            test.loc[i, "product_description"] = utils.lemmatize(
                row["product_description"]
            )
            test.loc[i, "query"] = utils.lemmatize(row["query"])
        test.to_csv("preprocessed_data/test_lemmatized.tsv", sep="\t")
    print("Lemmatization done")

    # drop NaN values
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    # removing extra spaces
    train["product_description"] = train["product_description"].str.replace(
        r"[^\w\s]", ""
    )
    train["query"] = train["query"].str.replace(r"[^\w\s]", "")
    test["product_description"] = test["product_description"].str.replace(
        r"[^\w\s]", ""
    )
    test["query"] = test["query"].str.replace(r"[^\w\s]", "")

    # removing digits
    train["product_description"] = train["product_description"].apply(
        lambda x: " ".join(x for x in x.split() if not x.isdigit())
    )
    train["query"] = train["query"].apply(
        lambda x: " ".join(x for x in x.split() if not x.isdigit())
    )
    test["product_description"] = test["product_description"].apply(
        lambda x: " ".join(x for x in x.split() if not x.isdigit())
    )
    test["query"] = test["query"].apply(
        lambda x: " ".join(x for x in x.split() if not x.isdigit())
    )

    # create bag of words
    utils.bag_of_words(train["product_description"], test=test["product_description"])
    print("BOW done")

    # create tfidf
    utils.tf_idf(train["product_description"], test=test["product_description"])
    print("TFIDF done")

    # create word2vec
    utils.word_vector(train, test=test, features=["product_description", "query"])
    utils.training_testing_split(
        "preprocessed_data/train_w2v.tsv",
        "median_relevance",
        drop_features=[
            "Unnamed: 0",
            "Unnamed: 0.1",
            "id",
            "product_title",
            "relevance_variance",
            "median_relevance",
            "query",
            "product_description",
        ],
    )
    print("Word2Vec done")

    # create vec using Glove embeddings
    utils.glove_embeddings(train, test=test, features=["product_description", "query"])
    print("Glove done")

    # create vec using fasttext
    utils.fasttext_embeddings(
        train, test=test, features=["product_description", "query"]
    )
    print("Fasttext done")