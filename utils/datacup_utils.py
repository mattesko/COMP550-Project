from sklearn.metrics import f1_score

def score1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') 