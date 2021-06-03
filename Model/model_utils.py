import re
import os


import pandas as pd


from nltk.util import ngrams

file_path = "/Users/eugenernst/PycharmProjects/Challenges_im_SCM/DataAnalysis/C-SCM-DATA-Candidates_Evaluation_Anonymized_SS21.xlsx"


def create_n_grams(data=None, n=3):
    if data is None:
        data = pd.read_excel(file_path, engine='openpyxl')
        data.dropna(axis=0, subset=['Evaluation Statement'], inplace=True)

    evalu_statements = data[['Person ID', 'Evaluation Statement']]

    token_dict = {}
    n_gram_dict = {}
    for row in evalu_statements.iterrows():
        row_index = row[0]
        data_row = row[1]

        eval_text = data_row['Evaluation Statement']
        pers_id = data_row['Person ID']

        try:
            eval_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', eval_text)
            tokens = [token for token in eval_text.split(" ") if token != ""]

            token_dict[pers_id] = tokens
            n_gram_dict[pers_id] = list(ngrams(tokens, n))

        except:
            print("error: \n")
            print(eval_text)

    return token_dict, n_gram_dict


if __name__ == "__main__":
    token_dict, grma_dict = create_n_grams()

