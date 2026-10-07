"""Train-only imputation, encoding and numeric standardization."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessor(d, columns):
    numeric = d[columns].select_dtypes('number').columns.tolist()
    categorical = [c for c in columns if c not in numeric]
    return ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)), categorical)],
        verbose_feature_names_out=False)
