import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

seed = 123
np.random.seed(seed)


PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DATA_FILEPATH = os.path.join(DATA_DIR, 'metadata_articles_dataframe.pkl')

data = pd.read_pickle(DATA_FILEPATH)

X_test, y_test = data[data["fold"] == "test"].drop(columns="label"), data[data["fold"] == "test"]["label"]

model_names = ["logistic_reg_raw", "SVM_raw", "NB_raw"]
predictions = []

for model_name in model_names:
    loaded_pred = pd.read_pickle(os.path.join(PROJECT_DIR, "predictions", model_name + ".pkl"))
    loaded_pred.columns = [model_name]
    predictions.append(loaded_pred)

    print("==========> Confusion matrix for model: " + model_name)
    print(confusion_matrix(y_test, loaded_pred))
    print("==========> By class F1 score for model: " + model_name)
    print(f1_score(y_test, loaded_pred, average=None))
    print("==========> Micro avg F1 score for model: " + model_name)
    print(f1_score(y_test, loaded_pred, average="micro"))
