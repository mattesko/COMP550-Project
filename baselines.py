import os

import pandas as pd
import matplotlib as mlab
import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse

import sklearn
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

import multiprocessing

# if you want to suppress warnings 
import warnings
warnings.filterwarnings("ignore")

def main():
    PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    FEATURE_COLUMN = 'X'
    LABEL_COLUMN = 'label'
    
    data_filepath = os.path.join(DATA_DIR, 'training_dataframe.pkl')
    X = pd.read_pickle(data_filepath)
    X, y = (X[FEATURE_COLUMN].values, X[LABEL_COLUMN].values)
    X = extract_TFIDF(X)
    splits = split(X, y)
    print('Without Preprocessing')
    test(splits)

    data_filepath = os.path.join(DATA_DIR, 'preprocessed_training_dataframe.pkl')
    X = pd.read_pickle(data_filepath)
    X, y = (X[FEATURE_COLUMN].values, X[LABEL_COLUMN].values)
    X = extract_TFIDF(X)
    print('With Preprocessing')
    splits = split(X, y)
    test(splits)


def extract_TFIDF(X):
    feature_pipeline = Pipeline([
            ('vect', text.TfidfVectorizer()),
            ('norm', preprocessing.Normalizer())        
    ])

    features = feature_pipeline.fit_transform(X)
    return features


def split(X, y):
    TEST_SIZE_SPLIT = 0.3
    RANDOM_STATE_SPLIT = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=TEST_SIZE_SPLIT, random_state=RANDOM_STATE_SPLIT)

    VALIDATION_SIZE_SPLIT = 0.5
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, stratify=y_test,
                                                        test_size=VALIDATION_SIZE_SPLIT, random_state=RANDOM_STATE_SPLIT)

    try:
        X_train_val = np.concatenate((X_train, X_val), axis=0)
    except Exception:
        X_train_val = sparse.vstack((X_train, X_val))

    y_train_val = np.concatenate((y_train, y_val), axis=0)

    splits = {
        'X_train': X_train,
        'X_train_val': X_train_val,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_train_val': y_train_val,
        'y_val': y_val,
        'y_test': y_test}

    return splits
    

def test(splits):

    # #### Logistic Regression
    # logistic_regression = LogisticRegression()
    # params = {
    #     'penalty': ['l1', 'l2'],
    #     'C': [1, 2, 5]}

    K_FOLD = 3
    # logistic_regression = fine_tune(logistic_regression, splits, params, K_FOLD)

    # #### Linear SVM
    # svm = SVC()
    # SVM_KERNEL = "linear"
    # params = {
    #     'kernel': [SVM_KERNEL],
    #     'C': [1, 5, 10]}

    # svm = fine_tune(svm, splits, params, K_FOLD)

    # #### Naive Bayes
    # naive_bayes = MultinomialNB()
    # params = {'alpha': [0.25, 0.5, 0.75, 1.0]}

    # naive_bayes = fine_tune(naive_bayes, splits, params, K_FOLD)

    # #### Ensemble Random Forest
    random_forest = RandomForestClassifier()
    params = {'n_estimators': [100, 200, 500]}

    random_forest = fine_tune(random_forest, splits, params, K_FOLD)

    # #### Random
    RANDOM_CLASSIFIER_RANDOM_STATE = 42
    RANDOM_CLASSIFIER_STRATEGY = "most_frequent"

    # random_classifier = DummyClassifier(strategy=RANDOM_CLASSIFIER_STRATEGY, random_state=RANDOM_CLASSIFIER_RANDOM_STATE)
    
    # random_classifier.fit(splits['X_train_val'], splits['y_train_val'])

    # y_test = splits['y_test']
    # y_pred = random_classifier.predict(splits['X_test'])
    # print(f'Random: \n{classification_report(y_test, y_pred)}')


def fine_tune(estimator, splits, params, k_fold):
    X_train = splits['X_train']
    X_train_val = splits['X_train_val']
    X_val = splits['X_val']
    X_test = splits['X_test']

    y_train = splits['y_train']
    y_train_val = splits['y_train_val']
    y_val = splits['y_val']
    y_test = splits['y_test']

    num_workers = multiprocessing.cpu_count() // 2
    grid_cv = GridSearchCV(estimator, params, cv=k_fold, verbose=1, n_jobs=num_workers)
    grid_cv.fit(X_train, y_train)
    
    y_pred = grid_cv.predict(X_val)
    print(f'{estimator.__class__}: \n{classification_report(y_val, y_pred)}')
    print(f'Best Params: {grid_cv.best_params_}')

    estimator.set_params(**grid_cv.best_params_)
    estimator.fit(X_train_val, y_train_val)
    print(f'{estimator.__class__} on Test Set: \n{classification_report(y_test, estimator.predict(X_test.toarray()))}')

    return estimator


if __name__ == '__main__':
    main()
    