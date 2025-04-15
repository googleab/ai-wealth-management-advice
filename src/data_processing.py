import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """Creates interaction features using DataFrame."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        X = X.copy()
        X['income_to_AUM'] = X['income'] / (X['AUM'] + 1)
        X['age_engagement'] = X['age'] * X['engagement_score']
        return X

def build_preprocessing_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Risk_Label'])
    y = df['Risk_Label']

    # step 1: firstly create interaction features while X is still a df
    feature_engineer = InteractionFeatureCreator()
    X = feature_engineer.fit_transform(X)

    # step 2: applying sklearn preprocessing ( so that it can now can be an array)
    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)

    return X_processed, y, pipeline
