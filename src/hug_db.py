# HugDB
import faiss
import numpy as np

class HugDB:

    def __init__(self):
        self.index = None
        self.id_to_data = {}
    
    def build_index(self, data_vectors):
        data_vectors = np.array(data_vectors).astype('float32')
        self.index = faiss.IndexFlatL2(data_vectors.shape[1])
        self.index.add(data_vectors)

    def add_item(self, item_id, item_vector):
        item_vector = np.array(item_vector).astype('float32').reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatL2(item_vector.shape[1])
        self.index.add(item_vector)
        self.id_to_data[item_id] = item_vector
    
    def search(self, query_vector, k=10):
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        item_ids = [self.id_to_data[i] for i in indices[0]]
        return distances[0], item_ids