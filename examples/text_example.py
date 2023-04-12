import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hug_db import HugDB
from utils import hugprocess_text, text_to_hugvector
from gensim.models import KeyedVectors
from cryptography.fernet import Fernet


file_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(file_root, "models", "GoogleNews-vectors-negative300.bin")

model = KeyedVectors.load_word2vec_format (model_path, binary=True)

data = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", 
    "d diam maecenas ultricies mi eget.",
    "Pellentesque habitant morbi tristique senectus et netus et malesuada fames.",
    "Praesent tristique magna sit amet purus gravida quis blandit turpis.",
    "Sit amet tellus cras adipiscing enim eu turpis egestas pretium."
]
preprocessed_data = [hugprocess_text(d) for d in data]
vectors = [text_to_hugvector(tokens, model) for tokens in preprocessed_data if tokens]

key = Fernet.generate_key()

hug_db = HugDB(encryption_key=key) #update code; self key generation should be handled by HugDB there should be the capabilitie to use a keychain system for diferent sets of data.

for i, vector in enumerate(vectors[:-1]):
    hug_db.add_item(str(i), vector)

query_vector = vectors[-1]
distances, results = hug_db.search(query_vector, k=2)

print("Search results:", results)