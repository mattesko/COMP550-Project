from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.metrics import classification_report

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

seed = 123
np.random.seed(seed)
MY_HOME = 'drive/My Drive/'

def to_tdidf(pickle_path="tdidf_list.pkl"):
    print("==========> Data Preprocessing")
    data = pd.read_pickle(os.path.join(MY_HOME, "preprocessed_training_dataframe.pkl"))

    print(data.head())

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['X'][0:2000])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    dense_arr = np.array(denselist)
    fp = os.path.join(MY_HOME, pickle_path)
    with open(fp, 'wb') as jar:
        pickle.dump(dense_arr, jar, protocol=pickle.HIGHEST_PROTOCOL)

def fc(denselist, data):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50), random_state=1)
    clf.fit(denselist[0:1500], data['label'][0:1500])

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    to_tdidf()
    with open(os.path.join(MY_HOME,"tdidf_list.pkl"), 'rb') as jar:
        dense_arr = pickle.load(jar)

    print(dense_arr)
    data = pd.read_pickle(os.path.join(MY_HOME, "preprocessed_training_dataframe.pkl"))
    print(data.shape)
    fc(dense_arr, data)
