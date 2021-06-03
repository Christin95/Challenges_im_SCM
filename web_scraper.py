import re

from requests import get
from bs4 import BeautifulSoup as bs
from py_thesaurus import Thesaurus as ts
from PyDictionary import PyDictionary


leadership_trait_list_url = 'https://briandownard.com/leadership-skills-list/'


def get_leaership_trait_list():
    res = get(leadership_trait_list_url)
    bs_html = bs(res.text, 'html.parser')
    h2_list = bs_html.find_all('h2')

    trait_list =[]
    for element in h2_list:
        element_string = str(element)
        element_string = remove_tags(element_string)

        if element_string[-1] == ":":
            element_string = remove_number(element_string)
            trait_list.append(element_string)

    return trait_list


def remove_tags(element, tag_list=['<h2>', '</h2>', '<strong>', '</strong>', '\xa0', '&amp']):
    for tag in tag_list:
        element = element.replace(tag, '')
    return element


def remove_number(element):
    element = element.split(".")[-1]
    element = element.split(":")[0]

    return element



def get_trait_synonym_list():
    dictionary = PyDictionary()

    trait_list = get_leaership_trait_list()

    trait = trait_list[0]

    dictionary.synonym(trait)

    trait_synonym_dict = {trait: dictionary.synonym(trait) for trait in trait_list}

    return trait_synonym_dict



if __name__ == "__main__":
    trait_dict = get_trait_synonym_list()

    for key in trait_dict.keys():
        print(" \n\n" + key + "   SYNONYMS:")
        print(trait_dict[key])


    #get_leaership_trait_list()