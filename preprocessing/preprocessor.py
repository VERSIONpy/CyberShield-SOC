"""
Stub for preprocessing.preprocessor to allow loading the pickled UNSWPreprocessor.
"""
import numpy as np
import pandas as pd


class UNSWPreprocessor:
    """
    Custom preprocessor for UNSW-NB15 dataset.
    Wraps a sklearn ColumnTransformer with numeric scaling + one-hot encoding.
    """
    
    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.feature_columns = []
        self.transformed_feature_names = []
        self.column_transformer = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.column_transformer is None:
            raise RuntimeError("Preprocessor not fitted")
        
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return self.column_transformer.transform(df[self.feature_columns])
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self):
        return np.array(self.transformed_feature_names)
    
    def __repr__(self):
        return f"UNSWPreprocessor(num_cols={len(self.num_cols)}, cat_cols={len(self.cat_cols)})"
