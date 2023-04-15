# examples/text_example.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hug_db import HugDB
from hug_db_load_balancer import HugDBLoadBalancer
from utils import hugprocess_text, text_to_hugvector
from gensim.models import KeyedVectors
import random

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

additional_data = [
    "Neque laoreet suspendisse interdum consectetur libero id faucibus nisl tincidunt.",
    "Massa sed elementum tempus egestas sed sed risus pretium.",
    "A condimentum vitae sapien pellentesque habitant morbi tristique.",
    "it amet risus nullam eget felis eget."
] * 100

random.shuffle(additional_data)
data = data + additional_data

preprocessed_data = [hugprocess_text(d) for d in data]
vectors = [text_to_hugvector(tokens, model) for tokens in preprocessed_data if tokens]

metadata_list = [
    {"category": "A", "length": "short"},
    {"category": "B", "length": "medium"},
    {"category": "A", "length": "long"},
    {"category": "C", "length": "short"}
] * 10

hug_db_replicas = [HugDB(encryption_key=None, nlist=1, M=6, num_shards=1) for _ in range(3)]

expanded_vector = vectors * 10
expanded_metadata = metadata_list * 10

for replica in hug_db_replicas:
    replica.build_index((vectors * 10)[:-1], metadata_list)

query_vector = vectors[-1][1]
metada_filter = lambda metadata: metadata["category"] == "A" and metadata["length"] == "short"
distances, results = hug_db_replicas[0].search(query_vector.reshape(1, -1), k=2, metadata_filter=metada_filter)

print("Search results:", results)