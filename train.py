import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils

if __name__ == "__main__":
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    X_train = pd.read_csv(trainfile, sep="\t")
    X_test = pd.read_csv(testfile, sep="\t")

    y_train = X_train["label"]
    X_train.drop(["label", "Unnamed: 0"], axis=1, inplace=True)

    y_test = X_test["label"]
    X_test.drop(["label", "Unnamed: 0"], axis=1, inplace=True)

    models = {
        "LogisticRegression": LogisticRegression(
            solver="newton-cg", max_iter=100, verbose=1
        ),
        "RidgeClassifier": RidgeClassifier(),
        "SGDClassifier": SGDClassifier(),
        "Perceptron": Perceptron(),
        "SVC": SVC(kernel="poly"),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=9),
        "RadiusNeighborsClassifier": RadiusNeighborsClassifier(radius=5.0),
        "GaussianProcessClassifier": GaussianProcessClassifier(),
        "GaussianNB": GaussianNB(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "ExtraTreesClassifier": ExtraTreesClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "MLPClassifier": MLPClassifier(activation="identity", solver="adam"),
    }

    mcc = {}
    for name, model in models.items():
        mcc[name] = utils.model_training(X_train, X_test, y_train, y_test, model)

    print(mcc)