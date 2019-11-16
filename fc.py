from sklearn.neural_network import MLPClassifier
import os
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

seed = 123
np.random.seed(seed)


def to_tdidf(pickle_path="tdidf_list.pkl"):
    print("==========> Data Preprocessing")
    data = pd.read_pickle("preprocessed_training_dataframe.pkl")

    print(data.head())

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['X'][0:200])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    dense_arr = np.array(denselist)

    with open(pickle_path, 'wb') as jar:
        pickle.dump(dense_arr, jar, protocol=pickle.HIGHEST_PROTOCOL)

def fc(denselist, data):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50), random_state=1)
    clf.fit(denselist, data['label'][0:200])

    print(clf.score(denselist[0:10], data['label'][0:10]))


if __name__ == '__main__':
    to_tdidf()
    with open("tdidf_list.pkl", 'rb') as jar:
        dense_arr = pickle.load(jar)

    print(dense_arr)
    data = pd.read_pickle("preprocessed_training_dataframe.pkl")
    fc(dense_arr, data)


