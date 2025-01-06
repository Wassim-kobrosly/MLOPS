from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class CustomModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def predict(self, X):
        return self.transform(X)
