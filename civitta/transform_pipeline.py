from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import miceforest as mf
import pandas as pd


@dataclass
class TransformationPipeline:
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(dtype=int))
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    numerical_power_pipeline = Pipeline([
        ("transform", PowerTransformer(standardize=False)),
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler())
    ])

    # preprocessing_pipeline = ColumnTransformer([
    #     ("nominal_preprocessor", nominal_pipeline, nominal_features),
    #     ("ordinal_preprocessor", ordinal_pipeline, ordinal_features),
    #     ("numerical_preprocessor", numerical_pipeline, numerical_features)
    # ])