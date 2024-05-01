import pandas as pd
import miceforest as mf


class DataImputer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def impute_miceforest(self):
        kds = mf.ImputationKernel(self.df, save_all_iterations=True, random_state=123)
        kds.mice(2)
        df_imputed = kds.complete_data()
        return df_imputed

