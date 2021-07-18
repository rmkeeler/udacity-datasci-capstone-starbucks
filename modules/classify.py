from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

import numpy as np

def train_multioutput(X_train, y_train):
    """
    Set up all of the steps required to train a model on the users and transactions datasets.
    Response rates are best when aggregated by offer_type, not when given for each offer.
    This is because data set is more balanced (45% response rates rather than 15% rates).

    train_test_split, then instanstiate some estimators, then GridSearchCV
    with multioutputclassifier.
    """
    return model

def score_multioutput(y_pred, y_test, labels):
    """
    Predict y_pred with X_test.
    Then, get confusion matrices for multioutput.
    Then, use confusion matrices, y_pred and y_test to produce:
    1. Accuracy
    2. Precision
    3. Recall

    input:
        y_pred: (array-like) predictions of y_test using X_test
        y_test: (array-like) actual observed values of y in the test set
        labels: list of labels of each multioutput target variable

    output:
        scores: dictionary mapping each target variable to accuracy, precision, recall
    """
    scores = dict()

    for i in range(len(labels)):
        # confusion matrix yields array with actuals in rows, predictions in columns
        matrix = confusion_matrix(y_test[:,i], y_pred[:,i])

        true_pos = matrix[1,1]
        false_pos = matrix[0,1]
        true_neg = matrix[0,0]
        false_neg = matrix[1,0]

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        scores[labels[i]] = dict(precision = precision,
                                recall = recall)

    return scores
