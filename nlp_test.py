import os
import nltk
import spacy

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import spatial

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

language_model = "en_core_web_lg"

try:
    nlp = spacy.load(language_model)
except:
    # download language model, if it is not already available
    dl_command = "python -m spacy download {}".format(language_model)
    os.system(dl_command)
    nlp = spacy.load(language_model)


# download Vader for nltk
nltk.download('vader_lexicon')

from Data.data_loader import load_data
data = load_data()
eval_data = data[['Person ID', 'Evaluation Statement']].dropna(axis=0)

print(len(eval_data))


statements_clean = eval_data.values.tolist()


creative_word_list = ['creative', 'constructive', 'resourceful', 'imaginative', 'ingenious', 'canny', 'inventive',
                      'full of ideas', 'clever', 'adventurous', 'innovative', 'originative', 'visionary', 'fanciful',
                      'forward thinker', 'pioneering', 'fertile', 'mastermind', 'genius', 'go-ahead', 'witty',
                      'eccentrically', 'inspiring', 'stimulating', 'encouraging', 'rich in ideas', 'inspirational']

open_word_list = ['open', 'outgoing', 'curious', 'open-minded', 'broad-minded', 'honest', 'empathetic', 'respectful',
                  'positivity', 'emotional intelleligence', 'interest', 'interested', 'adapting', 'informative',
                  'sharing', 'feedback', 'honesty', 'trust', 'valuing', 'diversity', 'perspective']

responsible_word_list = ['responsible', 'decisions', 'decision-maker', 'supportive', 'prepared', 'proactive',
                         'reliable', 'trustworthy', 'discipline', 'respectable', 'committed', 'integrity', 'pushing',
                         'assertive', 'obligated', 'judicious', 'organized', 'managing', 'consistent']

entrepreneurial_word_list = ['entrepreneurial', 'enterprising', 'entrepreneurially', 'profit-oriented', 'for-profit',
                             'profit-seeking', 'need for achievement', "self-efficacy", 'innovativeness',
                             'stress tolerant', 'need for autonomy', 'proactive', 'disruptive', 'personality',
                             'venturesome', 'prepared to take risks', 'visionary', 'goal-oriented', 'purposeful',
                             'active', 'engaged', 'maker', 'doer', 'self-starter', 'calm', 'passionate', 'positive',
                             'convinced']


cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']


# part of speech tag for adjectives
POS = {"ADJ"}

# threshold for the word similarity
# values below the threshold will be set to zero in the final evaluation
similarity_threshold = 0.2

eval_res_dict = {}
def create_similarity_set(i, sent, word_list, similarity_threshold=similarity_threshold):
    sim_set = set()
    for word in word_list:
        sim = cosine_similarity(i.vector, nlp(word).vector)
        if (sim >= similarity_threshold):
            sim_set.add(sent)

    return sim_set

person_id, statement = statements_clean[0]

def statement_core_evaluation(person_id, statement):
    adj_nouns_list = []
    if len(statement) != 0:
        for sent in nlp(statement).sents:
            for t in sent.as_doc():
                if (t.pos_ in POS):
                    adj_nouns_list.append((t, sent, person_id))

            for noun in sent.noun_chunks:
                adj_nouns_list.append((noun, sent, person_id))

        creative_set = set()
        open_set = set()
        responsible_set = set()
        entr_set = set()

        for i, sent, id_ in tqdm(adj_nouns_list):

            open_set.update(create_similarity_set(i, sent, open_word_list))

            responsible_set.update(create_similarity_set(i, sent, responsible_word_list))

            entr_set.update(create_similarity_set(i, sent, entrepreneurial_word_list))

            creative_set.update(create_similarity_set(i, sent, creative_word_list))


        ls_compound_creative = []
        if not creative_set:
            ls_compound_creative.append(0)
        else:
            for s in creative_set:
                ls_compound_creative.append(sentiment_scores(s.text))


        ls_compound_open = []
        if not open_set:
            ls_compound_open.append(0)
        else:
            for s in open_set:
                ls_compound_open.append(sentiment_scores(s.text))


        ls_compound_responsible = []
        if not responsible_set:
            ls_compound_responsible.append(0)
        else:
            for s in responsible_set:
                ls_compound_responsible.append(sentiment_scores(s.text))


        ls_compound_entr = []
        if not entr_set:
            ls_compound_entr.append(0)
        else:
            for s in entr_set:
                ls_compound_entr.append(sentiment_scores(s.text))


        eval_res_dict[person_id] = {"creative": np.mean(ls_compound_creative),
                                    "open": np.mean(ls_compound_open),
                                    "responsible": np.mean(ls_compound_responsible),
                                    "entrepreneurial": np.mean(ls_compound_entr)
                                    }

    else:
        eval_res_dict[person_id] = {"creative": 0,
                                    "open": 0,
                                    "responsible": 0,
                                    "entrepreneurial": 0
                                    }

    return eval_res_dict


eval_dict ={}
for person_id, statement in tqdm(statements_clean):
    eval_dict.update(statement_core_evaluation(person_id, statement))

print(eval_dict)






