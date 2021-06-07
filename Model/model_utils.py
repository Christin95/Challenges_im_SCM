import re
import os


import nltk
import geonamescache

import pandas as pd

from tqdm import tqdm
from nltk.util import ngrams
from geonamescache.mappers import country


from DataAnalysis.data_loader import load_data


stopwords = nltk.corpus.stopwords.words("english")

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


def get_eval_text(data=None, merge=True, num_statements=None, remove_stop_words=True, remove_numbers=True):
    if data is None:
        data = pd.read_excel(file_path, engine='openpyxl')
        data.dropna(axis=0, subset=['Evaluation Statement'], inplace=True)

    if num_statements is not None:
        data = data.iloc[:num_statements]

    eval_statements = data['Evaluation Statement'].values.tolist()

    if remove_stop_words:
        eval_statements = filter_stop_words(eval_statements)

    if remove_numbers:
        eval_statements = filter_numbers(eval_statements)


    if merge:
        evaluation_text = " ".join(eval_statements)


    return evaluation_text


def get_country_names(text=None):
    if text is None:
        text = get_eval_text()

    gc = geonamescache.GeonamesCache()
    country_names = list(gc.get_countries_by_names().keys())

    countries_in_text = [county for county in tqdm(country_names, desc="Scanning country names within evaluation statements: ") if county.lower() in text.lower()]

    return countries_in_text


def get_continents_names(text=None):
    if text is None:
        text = get_eval_text()

    gc = geonamescache.GeonamesCache()
    continent = gc.get_continents()
    continent_keys = list(continent.keys())
    key = continent_keys[0]
    continent[key]['name']
    continent_names = [continent[key]['name'] for key in tqdm(continent_keys, desc="Scanning continent names within evaluation statements: ") if continent[key]['name'].lower() in text.lower()]

    return continent_names


def get_city_names(text=None):
    if text is None:
        text = get_eval_text()

    gc = geonamescache.GeonamesCache()

    #country_names = get_country_names(text)
    #country_iso3_mapper = country(from_key='name', to_key='geonameid')
    #country_iso3 = [country_iso3_mapper(country_name) for country_name in country_names]

    city_names_dict = gc.get_cities()

    city_names_keys = list(city_names_dict.keys())

    city_names_list = [city_names_dict[key]['name'] for key in tqdm(city_names_keys, desc="Scanning city names within evaluation statements: ") if city_names_dict[key]['name'] in text]

    return city_names_list



def add_geo_info_to_df(data=None):
    if data is None:
        data = pd.read_excel(file_path, engine='openpyxl')
        data.dropna(axis=0, subset=['Evaluation Statement'], inplace=True)

    continent_names = get_continents_names()
    country_names = get_country_names()
    city_names = get_city_names()

    data_columns = data.columns.tolist()

    add_continent_string = lambda s: ", ".join(geo_info for geo_info in continent_names if geo_info in s)
    continent_series = data['Evaluation Statement'].apply(add_continent_string)
    continent_series.column = ['Continent']

    add_country_string = lambda s: ", ".join(geo_info for geo_info in country_names if geo_info in s)
    country_series = data['Evaluation Statement'].apply(add_country_string)
    country_series.column = ['Country']

    add_city_string = lambda s: ", ".join(geo_info for geo_info in city_names if geo_info in s)
    city_series = data['Evaluation Statement'].apply(add_city_string)
    city_series.column = ['City']

    data_goe_info_added = pd.concat([data, continent_series, country_series, city_series], axis=1)

    data_columns = data_columns + ['Continent', 'Country', 'City']

    data_goe_info_added.columns = data_columns

    data_goe_info_added.to_csv("/Users/eugenernst/PycharmProjects/Challenges_im_SCM/DataAnalysis/evaluation_data_augmented.csv")

    print(data_goe_info_added.head(10))


    print('finish georaphic search')


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def filter_numbers(word_list):
    return [word for word in word_list if hasNumbers(word) is False]

def filter_stop_words(word_list):
    return [word for word in word_list if word.lower() not in stopwords]



if __name__ == "__main__":

    data = pd.read_csv("/Users/eugenernst/PycharmProjects/Challenges_im_SCM/DataAnalysis/evaluation_data_augmented.csv")
    add_geo_info_to_df()
