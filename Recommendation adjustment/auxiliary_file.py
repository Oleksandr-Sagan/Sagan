import math
import pandas as pd
import numpy as np
import nltk
import re, string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

def pre_processing(file_path):
    document = open(file_path, 'r', encoding='utf-8').read()
    
    cleaned_tokens = []
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    for token, tag in pos_tag(nltk.word_tokenize(document)):

        if (re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', token)) or (re.search(r'http\S+|www.\S+', token)) or (re.search(r'/+.*', token)) or (re.search(r'\.', token)) or (re.search(r'\W+', token)) or (re.search(r'\b\d+\b', token)) or (re.search(r'\b\w\b', token)):
            continue

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
   
        token = lemmatizer.lemmatize(token, pos)

        if token not in string.punctuation and token.lower() not in stop_words + ['“', '``', "'", "''"]:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens



def training_data_cnstructing(corpus, window_size=5):
    data_set = []
    data_set_training = []
    unique_words = set([item for sublist in corpus for item in sublist])

    one_hot_vectors = {word: [1 if w == word else 0 for w in unique_words] for word in unique_words}
    
    for doc in corpus:
        for index, word in enumerate(doc[:-window_size], window_size):
            context = doc[index-window_size : index]
            data_set.append((doc[index], context))

    index_dict = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for i, word in enumerate(unique_words)}
    
    for pair in data_set:
        data_set_training.append((np.array([index_dict[i] for i in pair[1]]), np.array(one_hot_vectors[pair[0]])))

    

    return data_set_training, index_to_word

def document_vector(word2vec_model, doc):

    doc = [word for word in doc if word in word2vec_model.keys()]
    return np.mean([word2vec_model[w] for w in doc], axis=0)