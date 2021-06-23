import os
import nltk


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))

# checks whether number is included
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# filters numbers (ids) from the statements
def filter_numbers(word_list):
    return [word for word in word_list if hasNumbers(word) is False]

def filter_stop_words(word_list):
    stopwords = nltk.corpus.stopwords.words("english")
    return [word for word in word_list if word.lower() not in stopwords]
