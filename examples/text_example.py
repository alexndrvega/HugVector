import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from gensim.models import KeyedVectors
from utils import hugprocess_text, text_to_hugvector
from hug_db import HugDB

file_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(file_root, "models", "GoogleNews-vectors-negative300.bin")

model = KeyedVectors.load_word2vec_format (model_path, binary=True, limit=50000)

data = ["This is an example sentence.", "Annother example text."]
preprocessed_data = [hugprocess_text(d) for d in data]
vectors = [text_to_hugvector(tokens, model) for tokens in preprocessed_data if tokens]

vectors = [v for v in vectors if v is not None]

print("Shapes of individual vectors:", [v.shape for v in vectors])

vectors = np.array(vectors, dtype=np.float32)

hug_db = HugDB()
hug_db.build_index(vectors)


query_text = "Find a similar sentence."
query_tokens = hugprocess_text(query_text)
query_vector = text_to_hugvector(query_tokens, model).reshape(1, -1)
results = hug_db.search(query_vector, k=2)

print("Search results: ", results)