import os
import nltk
import numpy as np
import pandas as pd
import sklearn as sk

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from preprocessing import preprocess, create_dataframe_for_training

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

seed = 123
np.random.seed(seed)

PREPROCESSING = False
REMOVE_STOP_WORDS = False
THRESHOLD_INFREQUENT_WORDS = 0.05
MODEL_NAMES = ["logistic_reg_raw", "SVM_raw", "NB_raw"]

if __name__ == '__main__':

    print("==========> Data Preprocessing")

    PROJECT_DIR = os.getcwd()
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    DATA_FILEPATH = os.path.join(DATA_DIR, 'metadata_articles_dataframe.pkl')

    data = pd.read_pickle(DATA_FILEPATH)
    data = data[:100]

    X_train, y_train = data[data["fold"]=="train"].drop(columns="label"), data[data["fold"]=="train"]["label"]
    X_dev, y_dev = data[data["fold"]=="development"].drop(columns="label"), data[data["fold"]=="development"]["label"]
    X_test, y_test = data[data["fold"]=="test"].drop(columns="label"), data[data["fold"]=="test"]["label"]

    print("Training size: " + str(X_train.shape[0]))
    print("Development size: " + str(X_dev.shape[0]))
    print("Testing size: " + str(X_test.shape[0]))


    def generate_feature_matrix(X_train, X_dev, X_test, preprocessing=False, remove_stopwords=False, min_df=1):
        if preprocessing:
            X_train = preprocess(X_train, remove_stopwords)
            X_dev = preprocess(X_dev, remove_stopwords)
            X_test = preprocess(X_test, remove_stopwords)

        X_train = create_dataframe_for_training(X_train)
        X_dev = create_dataframe_for_training(X_dev)
        X_test = create_dataframe_for_training(X_test)

        vectorizer = CountVectorizer(min_df=min_df)
        X_train_fe = vectorizer.fit_transform(X_train)
        X_valid_fe = vectorizer.transform(X_dev)
        X_test_fe = vectorizer.transform(X_test)

        return X_train_fe.toarray(), X_valid_fe.toarray(), X_test_fe.toarray()


    X_train_fe, X_dev_fe, X_test_fe = generate_feature_matrix(X_train, X_dev, X_test,
                                                              preprocessing=PREPROCESSING,
                                                              remove_stopwords=REMOVE_STOP_WORDS,
                                                              min_df=THRESHOLD_INFREQUENT_WORDS)
    X_modeling, y_modeling = np.append(X_train_fe, X_dev_fe, axis=0), np.append(y_train, y_dev, axis=0)

    print("Training feature matrix shape: " + str(X_train_fe.shape))
    print("Development feature matrix shape: " + str(X_dev_fe.shape))
    print("Testing feature matrix shape: " + str(X_test_fe.shape))

    print("==========> Logistic Regression Tuning")

    accuracy_results = []
    hp_list = [0.01, 0.1, 0.5, 1, 10]
    for c in hp_list:
        logistc_reg = LogisticRegression(C=c)
        logistc_reg.random_state = seed
        logistc_reg.fit(X_train_fe, y_train)
        accuracy_results.append(logistc_reg.score(X_dev_fe, y_dev))
    print("Minimum accuracy of {} reached with parameter value of c={}".format(np.max(accuracy_results),
                                                                               hp_list[np.argmax(accuracy_results)]))

    print("==========> Logistic Regression Final Model")

    logistc_reg_final = LogisticRegression(C=hp_list[np.argmax(accuracy_results)])
    logistc_reg_final.random_state = seed
    logistc_reg_final.fit(X_modeling, y_modeling)
    log_reg_pred = logistc_reg_final.predict(X_test_fe)
    print("Accuracy of final model on test set: {}".format(logistc_reg_final.score(X_test_fe, y_test)))

    print("==========> SVM Tuning")

    accuracy_results = []
    hp_list = [0.01, 0.1, 0.5, 1]
    for c in hp_list:
        linear_svm = svm.LinearSVC(C=c)
        linear_svm.random_state = seed
        linear_svm.fit(X_train_fe, y_train)
        accuracy_results.append(linear_svm.score(X_dev_fe, y_dev))
    print("Minimum accuracy of {} reached with parameter value of c={}".format(np.max(accuracy_results),
                                                                               hp_list[np.argmax(accuracy_results)]))
    print("==========> SVM Final Model")

    linear_svm_final = svm.LinearSVC(C=hp_list[np.argmax(accuracy_results)])
    linear_svm_final.random_state = seed
    linear_svm_final.fit(X_modeling, y_modeling)
    linear_svm_pred = linear_svm_final.predict(X_test_fe)
    print("Accuracy of final model on test set: {}".format(linear_svm_final.score(X_test_fe, y_test)))

    print("==========> Naive Bayes Tuning")

    accuracy_results = []
    hp_list = [1e-10, 1e-5, 0.001, 0.01, 0.1, 1, 10]
    for a in hp_list:
        naive_bayes = MultinomialNB(alpha=a)
        naive_bayes.random_state = seed
        naive_bayes.fit(X_train_fe, y_train)
        accuracy_results.append(naive_bayes.score(X_dev_fe, y_dev))
    print("Minimum accuracy of {} reached with parameter value of c={}".format(np.max(accuracy_results),
                                                                               hp_list[np.argmax(accuracy_results)]))
    print("==========> Naive Bayes Final Model")

    naive_bayes_final = MultinomialNB(alpha=hp_list[np.argmax(accuracy_results)])
    naive_bayes_final.random_state = seed
    naive_bayes_final.fit(X_modeling, y_modeling)
    naive_bayes_pred = naive_bayes_final.predict(X_test_fe)
    print("Accuracy of final model on test set: {}".format(naive_bayes_final.score(X_test_fe, y_test)))

    print("==========> Saving predictions on test set")
    model_objs = [log_reg_pred, linear_svm_pred, naive_bayes_pred]

    PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

    for model_name, model_obj in zip(MODEL_NAMES, model_objs):
        pd.DataFrame(model_obj).to_pickle(os.path.join(PROJECT_DIR, "predictions", model_name + ".pkl"))
