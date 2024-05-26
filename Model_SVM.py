from sklearn.svm import SVC
from sklearn.base import BaseEstimator


class SVMClassifierWrapper(BaseEstimator):
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", random_state=42):
        self.model = SVC(
            C=C, kernel=kernel, gamma=gamma, probability=True, random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
