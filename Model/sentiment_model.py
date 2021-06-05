

import spacy
import numpy


from spacy.vocab import Vocab
from transformers import pipeline
from DataAnalysis.data_loader import load_evaluations



def sentiment_analysis():
    eval_data = load_evaluations()
    classifier = pipeline('sentiment-analysis')

    sentiment_dict = []
    for (row_index, row) in eval_data.iterrows():
        sentiment = classifier(row['Evaluation Statement'])
        sentiment_dict[row['Person ID']] = sentiment
        type(sentiment)



def text_analysis():
    eval_data = load_evaluations()


    text = eval_data['Evaluation Statement'].tolist()


    sample_text = text[0]
    nlp = spacy.load("en_core_web_md")
    doc = nlp(sample_text)



def set_word_embedding():

    nlp_lg = spacy.load('en_core_web_lg')

    text = "This could be a sample text for the given task."

    doc = nlp_lg(text)

    word1 = doc[0]
    word2 = doc[1]

    noun_chunks = list(doc.noun_chunks)

    noun1 = noun_chunks[0]
    noun2 = noun_chunks[1]

    noun1.similarity(noun2)

    word1.similarity(word2)








if __name__ == "__main__":
    set_word_embedding()
    sentiment_analysis()