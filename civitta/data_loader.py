from dataclasses import dataclass
import pandas as pd


@dataclass
class DataLoader:
    df_client_information = pd.read_excel("data/client_information.xlsx", sheet_name=1)
    df_loan_information = pd.read_excel("data/loan_information.xlsx", sheet_name=1)
    df_loan_outcome_information = pd.read_excel(
        "data/loan_outcome_information.xlsx", sheet_name=1
    )

    def merge_datasets(self):
        df = self.df_loan_outcome_information.merge(
            self.df_loan_information, on="clientid"
        )
        df = df.merge(self.df_client_information, on="clientid")
        return df

    def load(self):
        df = self.merge_datasets()
        return df
