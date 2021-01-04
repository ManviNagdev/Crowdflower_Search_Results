import utils
from os import path, listdir, walk
import sys
import pandas as pd
from tqdm import tqdm
import getopt

if __name__ == "__main__":
    """
    input: --lemmatize=, filepath, extract_path (if not extract_here)
    """
    options, remainder = getopt.getopt(sys.argv[1:], "o:v", ["lemmatize="])
    # print("OPTIONS   :", options)
    lem = False
    for opt, arg in options:
        if opt in ("-l", "--lemmatize"):
            lem = arg
    filepath = remainder[0]
    # print(filepath)
    try:
        extract_path = remainder[1]
    except:
        extract_path = None
    utils.unzip_files(
        path=filepath,
        extract_to=extract_path,
        files_to_unzip=utils.list_files(directory=filepath, extension=".zip"),
    )
    # print(extract_path)
    # print("OUTPUT    :", lem)
    # print("REMAINING :", remainder)
    # load dataset
    train = pd.read_csv("D:/Machine Learning/Crowdflower_Search_Results/data/train.csv")
    test = pd.read_csv("D:/Machine Learning/Crowdflower_Search_Results/data/test.csv")

    # drop NaN values
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    # convert html text to normal text
    train["product_description"] = train["product_description"].apply(utils.html2text)
    test["product_description"] = test["product_description"].apply(utils.html2text)

    # remove stopwords
    for i, row in tqdm(train.iterrows()):
        train.loc[i, "product_description"] = utils.remove_stopwords_without_tokenize(
            row["product_description"]
        )
    train.to_csv(
        "D:/Machine Learning/Crowdflower_Search_Results/preprocessed_data/train_removed_stopwords.csv"
    )
    for i, row in tqdm(test.iterrows()):
        test.loc[i, "product_description"] = utils.remove_stopwords_without_tokenize(
            row["product_description"]
        )
    test.to_csv(
        "D:/Machine Learning/Crowdflower_Search_Results/preprocessed_data/test_removed_stopwords.csv"
    )

    # lemmatize
    if lem == "True":
        for i, row in tqdm(train.iterrows()):
            train.loc[i, "product_description"] = utils.lemmatize(
                row["product_description"]
            )
        train.to_csv(
            "D:/Machine Learning/Crowdflower_Search_Results/preprocessed_data/train_lemmatized.csv"
        )
        for i, row in tqdm(test.iterrows()):
            test.loc[i, "product_description"] = utils.lemmatize(
                row["product_description"]
            )
        test.to_csv(
            "D:/Machine Learning/Crowdflower_Search_Results/preprocessed_data/test_lemmatized.csv"
        )