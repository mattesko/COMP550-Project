from sklearn.neural_network import MLPClassifier
import os
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

seed = 123
np.random.seed(seed)

def tdidf(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    dense_arr = np.array(denselist)

    return dense_arr



def to_tdidf(pickle_path="tdidf_list_full.pkl"):
    """
    process clean text into TF IDF, then dump into a pickle
    :param pickle_path: pickle file name
    :return:
    """
    print("==========> Data Preprocessing")
    data = pd.read_pickle("preprocessed_training_dataframe.pkl")

    print(data.head())


    X_train, X_test, y_train, y_test = split_tfidf(data['X'][0:500], data['label'][0:500])
    X_train_tdidf = tdidf(X_train)
    X_test_tdidf = tdidf(X_test)


    del X_train
    del X_test

    # this poor woman's laptop doesn't have enough ram
    # keeps killing my process so I now pickle them separately

    with open("tdidf_list_x_train.pkl", 'wb') as jar:
        pickle.dump(X_train_tdidf, jar, protocol=pickle.HIGHEST_PROTOCOL)
    with open("tdidf_list_x_test.pkl", 'wb') as jar:
        pickle.dump(X_test_tdidf, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open("tdidf_list_y_train.pkl", 'wb') as jar:
        pickle.dump(y_train, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open("tdidf_list_y_test.pkl", 'wb') as jar:
        pickle.dump(y_test, jar, protocol=pickle.HIGHEST_PROTOCOL)


def split_tfidf(denselist, label):
    X = denselist
    y = label # data['label']# for debugging [0:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def fc(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=2e-3, hidden_layer_sizes=(100), random_state=1)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    #to_tdidf()

    with open("tdidf_list_x_train.pkl", 'rb') as jar:
        X_train = pickle.load(jar)


    with open("tdidf_list_x_test.pkl", 'rb') as jar:
        X_test = pickle.load(jar)

    with open("tdidf_list_y_train.pkl", 'rb') as jar:
        y_train = pickle.load(jar)

    with open("tdidf_list_y_test.pkl", 'rb') as jar:
        y_test = pickle.load(jar)

    fc(X_train, X_test, y_train, y_test)


