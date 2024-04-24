import pandas as pd


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    def drop_columns(self, columns: list):
        self.df.drop(columns=columns)

    def get_x_cols(self):
        pass

    def transform(self):
        df = self.drop_columns()
        return df
    


