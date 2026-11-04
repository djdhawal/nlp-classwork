"""Task 1: Logistic Regression on top of any feature matrix."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .config import SEED


def train_and_evaluate(F_train, y_train, F_test, y_test, C=1.0):
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(F_train, y_train)
    pred = clf.predict(F_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "clf": clf,
    }
