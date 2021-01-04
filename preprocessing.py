import utils
from os import path, listdir, walk
import sys
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    """
    input: filepath, extract_path (if not extract_here)
    """
    filepath = sys.argv[1]
    try:
        extract_path = sys.argv[2]
    except:
        extract_path = None
    utils.unzip_files(
        path=filepath,
        extract_to=extract_path,
        files_to_unzip=utils.list_files(directory=filepath, extension=".zip"),
    )

    # load dataset
    train = pd.read_csv("D:/Machine Learning/Crowdflower_Search_Results/data/train.csv")
    test = pd.read_csv("D:/Machine Learning/Crowdflower_Search_Results/data/test.csv")

    # drop NaN values
    train.dropna(inplace=True)

    # convert html text to normal text
    train["product_description"] = train["product_description"].apply(utils.html2text)
    print(train.shape, train.head(), sep="\n")
    for i, row in tqdm(train.iterrows()):
        train.loc[i, "product_description"] = utils.remove_stopwords_without_tokenize(
            row["product_description"]
        )
    train.to_csv(
        "D:/Machine Learning/Crowdflower_Search_Results/preprocessed_data/removed_stopwords.csv"
    )

    # lemmatize
    for i, row in tqdm(train.iterrows()):
        train.loc[i, "product_description"] = utils.lemmatize(
            row["product_description"]
        )
    train.to_csv(
        "D:/Machine Learning/Crowdflower_Search_Results/preprocessed_data/lemmatized.csv"
    )