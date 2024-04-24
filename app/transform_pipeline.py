from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


@dataclass
class TransformationPipeline(Pipeline):
    Pipeline()


numerical_features = [
    "ratio", "loan_initial_term", 
    "loan_initial_amount", "loan_to_value_ratio",
    "annual_percentage_rate", "monthly_interest_rate"
    ]
nominal_features = [
    "loan_type", "region", "branch", "client_gender"
    ]
ordinal_features = [
    "vehicle_production_year", "vehicle_initial_assessment_value", 
    "age", "had_car_loan", "had_other_loans"
]

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=True, handle_unknown="ignore"))
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessing_pipeline = ColumnTransformer([
    ("nominal_preprocessor", nominal_pipeline, nominal),
    ("ordinal_preprocessor", ordinal_pipeline, ordinal),
    ("numerical_preprocessor", numerical_pipeline, numerical)
])