import os
import spacy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from tqdm import tqdm
from spacy.tokens import Doc
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from transformers import pipeline


# own functions
from Data.data_loader import load_evaluations, load_data
from Data.web_scraper import get_leaership_trait_list
from Model.model_utils import get_eval_text, remove_punctuation, get_nlp_lg


# file path where modified_data will potetially be stored
save_file_path = "/Users/eugenernst/PycharmProjects/Challenges_im_SCM/Data/C-SCM-DATA-Candidates_Evaluation_trait_included_SS21.csv"

core_value_list = ['open', 'responsible', 'creative', 'entrepreneurial']
nlp_lg = get_nlp_lg()

core_value_doc = nlp_lg(" ".join(core_value_list))

def get_text_doc():
    """ """
    text = get_eval_text()
    text_doc = nlp_lg(text)
    return text_doc

def create_sentence_similarity_df(doc=None, trait_doc=core_value_doc):
    """

    Parameters
    ----------
    doc :
         (Default value = None)
    trait_doc :
         (Default value = core_value_doc)

    Returns
    -------

    """
    if doc is None:
        doc = get_text_doc()

    trait_similarity_df = pd.DataFrame()

    sentences = list(doc.sents)
    num_sent = len(sentences)

    # TODO: reverse the ordering of the for loops
    for trait in list(trait_doc):
        trait_list = []
        for i in range(num_sent):
            sent = sentences[i]
            trait_list.append([trait.similarity(sent), sent.text, sent.start, sent.end])


        single_trait_df = pd.DataFrame(trait_list)
        single_trait_df.columns = [trait.text + postfix for postfix in ["_sim", "_sent", "_start", "_end"]]

        # convoluted dataframe
        trait_similarity_df = pd.concat([trait_similarity_df, single_trait_df], axis=1)

    return trait_similarity_df


def mark_sentence_boundaries(doc, boundary="."):
    """

    Parameters
    ----------
    doc :
        
    boundary :
         (Default value = ".")

    Returns
    -------

    """
    indexes = []
    for i, token in enumerate(doc):
        if token.text == boundary:
            doc[i+1].sent_start = True
            indexes.append(token.i)

    np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    np_array = np.delete(np_array, indexes, axis=0)
    doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    return doc2


def get_leadership_trait_doc():
    """ """
    leadership_trait_list = get_leaership_trait_list()
    leadership_trait_string = "".join(leadership_trait_list)
    leadership_trait_string = remove_punctuation(leadership_trait_string)
    leadership_trait_doc_temp = nlp_lg(leadership_trait_string)

    return leadership_trait_doc_temp



def add_core_values_similarity(data=None, normalize=True, safe_file=True):
    """

    Parameters
    ----------
    data :
         (Default value = None)
    normalize :
         (Default value = True)
    safe_file :
         (Default value = True)

    Returns
    -------

    """
    if data is None:
        data = load_data()

    trait_similarity_list = []
    for row_index, row in tqdm(data.iterrows()):
        try:
            row_eval_statement = row['Evaluation Statement']
            row_doc = nlp_lg(row_eval_statement)

            # add similarity of different traits
            trait_similarity_list.append([trait.similarity(row_doc) for trait in core_value_doc])
        except:
            print('SIMILARITY DETERMINATION FAILED FOR: \n \t {}'.format(str(row_index)))
            trait_similarity_list.append([np.nan for trait in core_value_doc])

    trait_columns = [trait.text for trait in core_value_doc]
    trait_similarity_df = pd.DataFrame(trait_similarity_list, columns=trait_columns)

    if normalize:
        trait_similarity_df = normalize_columns(trait_similarity_df)

    trait_similarity_df.index = data.index

    data_modified = pd.concat([data, trait_similarity_df], axis=1)

    if safe_file:
        data_modified.to_csv("/Users/eugenernst/PycharmProjects/Challenges_im_SCM/Data/C-SCM-DATA-Candidates_Evaluation_trait_included_SS21.csv")

    return data_modified


def word_similarity_score(eval_string, word='open'):
    if type(eval_string) == list:
        output = np.mean([nlp_lg(word).similarity(nlp_lg(eval)) for eval in eval_string if eval not in  ["", " "]])
    else:
        output = nlp_lg(word).similarity(nlp_lg(eval_string))

    return output

def word_score(word='open', eval_string="hi is the most open guy iin the whole world"):
    """

    Parameters
    ----------
    word :
         (Default value = 'open')
    eval_string :
         (Default value = "hi is the most open guy iin the whole world")

    Returns
    -------

    """
    return nlp_lg(word).similarity(nlp_lg(eval_string))


def add_trait_values_similarities(data=None, normalize=True):
    """

    Parameters
    ----------
    data :
         (Default value = None)
    normalize :
         (Default value = True)

    Returns
    -------

    """
    if data is None:
        data = load_data()

    leadership_traits_doc = get_leadership_trait_doc()

    trait_similarity_list = []
    for row_index, row in data.iterrows():
        try:
            row_eval_statement = row['Evaluation Statement']
            row_doc = nlp_lg(row_eval_statement)

            # add similarity of different traits
            trait_similarity_list.append(leadership_traits_doc.similarity(row_doc))
        except:
            print('SIMILARITY DETERMINATION FAILED FOR: \n \t {}'.format(str(row_index)))
            trait_similarity_list.append([np.nan for trait in core_value_doc])

    trait_similarity_df = pd.DataFrame(trait_similarity_list, columns=['leadership traits'])

    if normalize:
        trait_similarity_df = normalize_columns(trait_similarity_df)

    trait_similarity_df.index = data.index

    data_modified = pd.concat([data, trait_similarity_df], axis=1)

    return data_modified

def standardize_columns(df, columns=None):
    """

    Parameters
    ----------
    df :
        
    columns :
         (Default value = None)

    Returns
    -------

    """
    # if no column is given, all columns will be standardize
    if columns is None:
        columns = df.columns

    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df


def normalize_columns(df, columns=None):
    """

    Parameters
    ----------
    df :
        
    columns :
         (Default value = None)

    Returns
    -------

    """
    # if no column is given, all columns will be standardize
    if columns is None:
        columns = df.columns

    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


def get_neg_token(doc):
    """

    Parameters
    ----------
    doc :
        

    Returns
    -------

    """
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
    negation_head_tokens = [token.head for token in negation_tokens]
    for token in negation_head_tokens:
        print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])


def get_doc_token_list(doc):
    """

    Parameters
    ----------
    doc :
        

    Returns
    -------

    """
    doc_token_list = [doc[k] for k in range(doc.__len__())]
    return doc_token_list


def df_add_empty_cols(data, cols=core_value_list):
    """

    Parameters
    ----------
    data :
        
    cols :
         (Default value = core_value_list)

    Returns
    -------

    """
    for col in cols:
        data[col] = 0
    return data


def sentiment_analysis():
    """ """
    eval_data = load_evaluations()
    classifier = pipeline('sentiment-analysis')

    sentiment_dict = []
    for (row_index, row) in eval_data.iterrows():
        sentiment = classifier(row['Evaluation Statement'])
        sentiment_dict[row['Person ID']] = sentiment
        type(sentiment)



if __name__ == "__main__":
    add_core_values_similarity()
    pass