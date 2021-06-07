import os
import pandas as pd


file_path = "/Users/eugenernst/PycharmProjects/Challenges_im_SCM/DataAnalysis/C-SCM-DATA-Candidates_Evaluation_Anonymized_SS21.xlsx"

def load_data():
    data = pd.read_excel(file_path, engine='openpyxl')

    return data


def load_evaluations():
    data = load_data()
    eval_statement_data = data[['Person ID', 'Evaluation Statement']]

    return eval_statement_data



if __name__ =="__main__":
    load_evaluations()




