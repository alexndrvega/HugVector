# src/utils.py

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from PIL import Image, ImageOps

nltk.download("punkt")
nltk.download("stopwords")
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# hugprocess_text
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


# hugprocess_img
def hugprocess_img(image_path, resize_dim=(224, 224), normalize=True, augment=False):
    img = Image.open(image_path)

    img = img.resize(resize_dim, Image.ANTIALIAS)

    if augment:
        img = ImageOps.mirror(img) if np.random.rand() > 0.5 else img
        rotation_angle =  np.random.uniform(-20, 20)
        img = img.rotate(rotation_angle)
    
    img_array = np.array(img)

    if normalize:
        img_array = img_array / 255.0
    
    return img_array

