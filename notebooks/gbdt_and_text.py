# /// script
# dependencies = [
#   "pandas==2.2.3",
#   "scikit-learn==1.6.1",
#   "xgboost==2.1.4"
# ]
# ///

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

class TextEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, text_column):
        self.text_column = text_column
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()

    def fit(self, X, y):
        text_data = X[self.text_column]
        X_tfidf = self.tfidf.fit_transform(text_data)

        self.model.fit(X_tfidf, y)

        return self

    def transform(self, X):
        text_data = X[self.text_column]

        X_tfidf = self.tfidf.transform(text_data)

        probs = self.model.predict_proba(X_tfidf)[:, 1]

        X_transformed = X.copy()
        X_transformed['text_model_prob'] = probs

        return X_transformed.drop(columns=[self.text_column])

def pipeline():
    return Pipeline([
        ('encode_text', TextEncoder("product_description")),
        ('classifier', XGBClassifier(enable_categorical=True))
    ])

    return pipeline


if __name__ == "__main__":
    df = pd.DataFrame({
        'product_description': ['High quality wireless earbuds', 'Basic t-shirt', 
                               'Premium smartphone with great camera'],
        'average_rating': [4.5, 3.2, 4.8],
        'price': [99.99, 19.99, 899.99],
        'Product Category': ['Electronics', 'Clothing', 'Electronics']
    })

    df['Product Category'] = df['Product Category'].astype('category')

    y = [1, 0, 1]  

    pipeline = pipeline()
    pipeline.fit(df, y)

    new_df = pd.DataFrame({
        'product_description': ['Bluetooth headphones with noise cancellation'],
        'average_rating': [4.3],
        'price': [149.99],
        'Product Category': ['Electronics']
    })

    new_df['Product Category'] = new_df['Product Category'].astype('category')

    click_prob = pipeline.predict_proba(new_df)[0, 1]
    print(f"Probability of click: {click_prob:.2f}")
