# src/utils.py

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("punkt")
nltk.download("stopwords")
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))


def hugprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens] 
    return tokens

def text_to_hugvector(tokens, model):
    token_vectors = [model[t] for t in tokens if t in model]
    if not token_vectors:
        return None
    return np.mean(token_vectors, axis=0)

# utils tests.
#sample_text = "This is an example sentence to demostrate text pre-processing."
#preprocessed_tokens = hugprocess_text(sample_text)
#print(preprocessed_tokens)
