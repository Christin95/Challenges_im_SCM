#%%

import shap
import transformers
import numpy as np
from Model.evaluation_model import word_similarity_score
import spacy

from Data.data_loader import load_data



language_model = "en_core_web_lg"
nlp = spacy.load(language_model)


open_word_list = ['open', 'outgoing', 'curious', 'open-minded', 'broad-minded', 'honest', 'empathetic', 'respectful',
                  'positivity', 'emotional intelleligence', 'interest', 'interested', 'adapting', 'informative',
                  'sharing', 'feedback', 'honesty', 'trust', 'valuing', 'diversity', 'perspective']


open_words_string = " ".join(open_word_list)



#%%

statement = "He is the most open person I have ever seen. However, he is not creative."


sent = list(nlp(statement).sents)
sent_doc = sent.as_doc()
adjectives = [t.text for t in sent.as_doc() if t.pos_ in ['ADJ']]

nouns = list(sent.noun_chunks)



def get_adjectives(statement):
    sent_list = list(nlp(statement).sents)
    adj_list = []
    for sent in sent_list:
        adj_list = adj_list + [t.text for t in sent.as_doc() if t.pos_ in ['ADJ']]
    return adj_list


adjectives = get_adjectives()

#%%

model = transformers.pipeline('sentiment-analysis', return_all_scores=True)
explainer = shap.Explainer(model)

statement = "He is the most open person I have ever seen. However, he is not creative."
def split_word_string(word_string=statement, pad=True):

    splited_word_list = word_string.split(" ")
    if pad:
        splited_word_list = [""] + [word +" " for word in splited_word_list] + [""]

    return splited_word_list

def modify_shap_value(explainer=explainer, statement=statement, sim_word=open_words_string):
    shap_values = explainer([statement])
    shap_value = shap_values[0, :, 'POSITIVE']

    modified_shap_value = shap_value
    splited_statement = split_word_string(statement)

    #modified_shap_value.base_values = word_similarity_score(splited_statement, word=sim_word)
    modified_shap_value.base_values = float(0)
    print(modified_shap_value.base_values)
    modified_shap_value.values = np.array([word_similarity_score(single_word, word=sim_word) for single_word in splited_statement])
    #modified_shap_value.hierarchical_values = np.array([word_similarity_score(single_word, word=sim_word) for single_word in splited_statement])
    modified_shap_value.data = np.array(splited_statement)

    return modified_shap_value

#%%
new_shapp_value = modify_shap_value(explainer,  statement,sim_word='creativity')
print(new_shapp_value)
#%%
new_shapp_value.hierarchical_values = new_shapp_value.values
print(new_shapp_value.hierarchical_values)
print(new_shapp_value.values)

#%%
shap_val = explainer(statement)

#%%
shap.plots.text(new_shapp_value, cmax=1)

#%%
#shap.plots.bar(new_shapp_value)

shap.plots.bar(new_shapp_value)


#%%


