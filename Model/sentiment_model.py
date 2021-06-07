import spacy
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')



from spacy.vocab import Vocab

from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage
from transformers import pipeline

from spacy.symbols import ADJ

from model_utils import get_eval_text
from DataAnalysis.data_loader import load_evaluations
from PyDictionary import PyDictionary
dictionary = PyDictionary()


# own functions
from DataAnalysis.data_loader import load_data

big_four_traits = ['open', 'responsible', 'creative', 'entrepreneurial']
nlp_lg = spacy.load('en_core_web_lg')
trait_doc = nlp_lg(" ".join(big_four_traits))


def extend_synonyms(word_list):
    synonym_set = set()
    for word in word_list:
        print(word)
        word = word.replace(" ", "-")
        try:
            a = set(dictionary.synonym(word)[:2])
            synonym_set = synonym_set.union(a)
        except:
            print('stopp')
    synonym_set = set(word_list).union(synonym_set)

    return list(synonym_set)





def get_big_four_synonym_dict():

    openness = ['open',  'openness', 'outgoing', 'curious', 'open-minded', 'broad-minded', 'honest', 'empathetic', 'respectful', 'positivity', 'emotional intelleligence', 'interest', 'interested', 'adapting' , 'informative', 'sharing', 'feedback', 'honesty', 'trust', 'valuing', 'diversity', 'perspective']
    openness = extend_synonyms(openness)

    responsible = ['responsible', 'decisions', 'decision-maker', 'supportive', 'prepared', 'proactive', 'reliable', 'trustworthy', 'discipline', 'respectable', 'committed', 'integrity', 'pushing', 'assertive', 'obligated', 'judicious', 'organized', 'managing', 'consistent']
    responsible = extend_synonyms(responsible)

    creative = ['creative', 'constructive', 'resourceful', 'imaginative', 'ingenious', 'canny', 'inventive', 'full of ideas', 'clever', 'adventurous', 'innovative', 'originative', 'visionary', 'fanciful', 'forward thinker', 'pioneering', 'fertile', 'mastermind', 'genius', 'go-ahead', 'witty', 'eccentrically', 'inspiring', 'stimulating', 'encouraging', 'full of ideas', 'rich in ideas', 'inspirational']
    creative = extend_synonyms(creative)

    entrepreneurial = ['entrepreneurial', 'enterprising', 'entrepreneurially', 'profit-oriented', 'for-profit', 'profit-seeking', 'need for achievement', 'self-efficacy', 'innovativeness', 'stress tolerant', 'need for autonomy', 'proactive', 'disruptive', 'personality', 'venturesome', 'prepared to take risks', 'visionary', 'goal-oriented', 'purposeful', 'active', 'engaged', 'maker', 'doer', 'self-starter', 'calm', 'passionate', 'positive', 'convinced']
    entrepreneurial = extend_synonyms(entrepreneurial)

    trait_synonym_dict = {'open': openness,
                          'responsible': responsible,
                          'creative': creative,
                          'entrepreneurial': entrepreneurial}
    return trait_synonym_dict


def get_big_four_synonym_simialrity():

    trait_dict = get_big_four_synonym_dict()

    nlp_lg = spacy.load('en_core_web_lg')

    text = get_eval_text(num_statements=50)

    data = load_data()


    evaluation_statements = data['Evaluation Statement']

    vector_data = {
        "open": np.random.uniform(-1, 1, (300,)),
        "responsible": np.random.uniform(-1, 1, (300,)),
        "creative": np.random.uniform(-1, 1, (300,)),
        "entrepreneurial": np.random.uniform(-1, 1, (300,))
    }

    trait_similarity = []
    for row, statement in evaluation_statements.iteritems():
        doc = nlp_lg(statement)
        doc.sentiment





    doc = nlp_lg(text)

    sentences = doc.get_lca_matrix()

    doc_vectors = np.concatenate([np.expand_dims(doc[k].vector, axis=-1) for k in range(doc.__len__())], axis=1).transpose()


    for ent in doc.ents:
        print(ent)





def visualize_word_embedding():
    nlp_lg = spacy.load('en_core_web_lg')
    language_model = SpacyLanguage(nlp_lg)

    text = get_eval_text(num_statements=2000)
    text = remove_punctuation(text)
    doc = nlp_lg(text.lower())

    adjectives = filter_adjectives(doc)

    embeddings = language_model[adjectives]
    embeddings.plot(kind="scatter", show_ops=True, x_axis='open', y_axis='creative', color="red")

    #embeddings.plot(kind="scatter", show_ops=True)
    #plt.grid()
    plt.show()






def create_word_embeddings():
    nlp_lg = spacy.load('en_core_web_lg')
    print('test')

    data = load_data()
    data = df_add_empty_cols(data, big_four_traits)


    nlp_lg = spacy.load('en_core_web_lg')
    text = get_eval_text()
    doc = nlp_lg(text)

    trait_doc1 = trait_doc[0]

    doc_token_list = get_doc_token_list(doc)
    sentences = list(doc.sents)
    n_chunks = list(doc.noun_chunks)


    sent1 = sentences[1]



def get_text_doc():
    nlp_lg = spacy.load('en_core_web_lg')
    text = get_eval_text()
    text_doc = nlp_lg(text)
    return text_doc


def create_sentence_similarity_df(doc=None, trait_doc=trait_doc):
    if doc is None:
        doc = get_text_doc()

    trait_similarity_df = pd.DataFrame()

    sentences = list(doc.sents)
    num_sent = len(sentences)

    for trait in list(trait_doc):
        trait_list = []
        for i in range(num_sent):
            sent = sentences[i]
            trait_list.append([trait.similarity(sent), sent.text, sent.start, sent.end])


        single_trait_df = pd.DataFrame(trait_list)
        single_trait_df.columns = [trait.text + postfix for postfix in ["_sim","_sent", "_start", "_end"]]

        # convoluted dataframe
        trait_similarity_df = pd.concat([trait_similarity_df, single_trait_df], axis=1)

    return trait_similarity_df




def get_neg_token(doc):
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
    negation_head_tokens = [token.head for token in negation_tokens]
    for token in negation_head_tokens:
        print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])



def christin_notebook():
    from spacy.lang.en.stop_words import STOP_WORDS
    import nltk
    nlp = spacy.load("en_core_web_lg")


    data = load_data()

    statements = data['Evaluation Statement'].tolist()

    text = "He is the most creative person in the WHOLE world."
    test = text + " " + statements[0]
    doc = nlp(test)

    import itertools as it
    import typing as tp

    VERB_POS = {"VERB", "AUX"}
    SPLIT_WORDS = {"and", "although", "but", "however", "except", "also", "nevertheless"}

    language_model = "en_core_web_lg"
    nlp = spacy.load(language_model)

    ls_sentences = []

    for sent in doc.sents:
        index = 0
        for t in sent.as_doc():
            if (t.text in SPLIT_WORDS or t.pos_ == "PUNCT"):
                ls_sentences.append(sent.as_doc()[index:t.i])
                index = t.i

    cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
    print(ls_sentences[0].text, cosine_similarity(ls_sentences[0].vector, nlp("creative").vector))

    from scipy import spatial


    open_ = ['outgoing', 'curious', 'open-minded', 'broad-minded', 'honest', 'empathetic', 'respectful', 'positivity',
             'emotional intelleligence', 'interest', 'interested', 'adapting', 'informative', 'sharing', 'feedback',
             'honesty', 'trust', 'valuing', 'diversity', 'perspective']
    responsible = ['decisions', 'decision-maker', 'supportive', 'prepared', 'proactive', 'reliable', 'trustworthy',
                   'discipline', 'respectable', 'committed', 'integrity', 'pushing', 'assertive', 'obligated',
                   'judicious', 'organized', 'managing', 'consistent']
    creative = ['creative', 'constructive', 'resourceful', 'imaginative', 'ingenious', 'canny', 'inventive',
                'full of ideas', 'clever', 'adventurous', 'innovative', 'originative', 'visionary', 'fanciful',
                'forward thinker', 'pioneering', 'fertile', 'mastermind', 'genius', 'go-ahead', 'witty',
                'eccentrically', 'inspiring', 'stimulating', 'encouraging', 'rich in ideas', 'inspirational']
    entrepreneurial = ['entrepreneurial', 'enterprising', 'entrepreneurially', 'profit-oriented', 'for-profit',
                       'profit-seeking', 'need for achievement', "self-efficacy", 'innovativeness', 'stress tolerant',
                       'need for autonomy', 'proactive', 'disruptive', 'personality', 'venturesome',
                       'prepared to take risks', 'visionary', 'goal-oriented', 'purposeful', 'active', 'engaged',
                       'maker', 'doer', 'self-starter', 'calm', 'passionate', 'positive', 'convinced']

    threshold = 0.725

    sent_creative = []
    set_creative = set()
    for l in ls_sentences:
        for c in creative:
            if (cosine_similarity(l.vector, nlp(c).vector) >= threshold):
                sent_creative.append((l, c, cosine_similarity(l.vector, nlp(c).vector)))
                set_creative.add(l)

    print(set_creative)




def get_doc_token_list(doc):
    doc_token_list = [doc[k] for k in range(doc.__len__())]
    return doc_token_list



def get_doc_lg(text=None):
    if text is None:
        text = get_eval_text()

    nlp_lg = spacy.load('en_core_web_lg')



def df_add_empty_cols(data, cols=big_four_traits):
    for col in cols:
        data[col] = 0
    return data


def filter_adjectives(doc):
    adjectives = list(set([word.text for word in doc if word.head.pos == ADJ]))
    return adjectives



def remove_punctuation(text):
    for symb in string.punctuation:
        text = text.replace(symb, "")

    return text





def create_onb_vectors(text_list=big_four_traits):
    num_words = len(text_list)
    random_vectors = np.random.randn(300, num_words)
    onb_basis, R = np.linalg.qr(random_vectors)


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q




def sentiment_analysis():
    eval_data = load_evaluations()
    classifier = pipeline('sentiment-analysis')

    sentiment_dict = []
    for (row_index, row) in eval_data.iterrows():
        sentiment = classifier(row['Evaluation Statement'])
        sentiment_dict[row['Person ID']] = sentiment
        type(sentiment)



if __name__ == "__main__":
    create_word_embeddings()
    #visualize_word_embedding()
    #get_big_four_synonym_simialrity()
    #sentiment_analysis()