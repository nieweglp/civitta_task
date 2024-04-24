import pandas as pd
from sklearn.model_selection import train_test_split


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self.seed = 123
        self.X = None
        self.y = None

    def drop_columns(self, columns: list):
        self.df.drop(columns=columns)

    def get_x_dataset(self, features):
        self.X = self.df[features]

    def get_y_dataset(self, target):
        self.y = self.df[target]

    def transform(self):
        df = self.drop_columns()
        return df

    def split_columns(self):
        return train_test_split(
            self.X, 
            self.y, 
            random_state=self.seed, 
            train_size=0.8
        )
                
    


