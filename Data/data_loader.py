import os
import pandas as pd


file_path = "Data/ETweb-Candidates_CORE Evaluation Text Analytics_Anonymized.xlsx"

file_path_processed = "Data/C-SCM-DATA-Candidates_Evaluation_trait_included_SS21.csv"


def load_data():
    data = pd.read_excel(file_path, engine='openpyxl', sheet_name=1)
    data = data[data['Evaluation Statement'].isna()==False]

    return data


def load_processed_data():
    data = pd.read_csv(file_path_processed, index_col=0)

    return data


def load_evaluations(columns=None):
    data = load_data()

    if columns is None:
        columns = data.columns

    eval_statement_data = data[columns]

    eval_statement_data.dropna(axis=0)

    return eval_statement_data



if __name__ =="__main__":
    load_data()




