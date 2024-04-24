import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train():
    model_pipeline = Pipeline([
        ("preprocessor", preprocessing_pipeline),
        ("random_forest", RandomForestClassifier)
    ])
    model_pipeline.fit()



if __name__ == "__main__":
    mlflow.autolog()
    train()