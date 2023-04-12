# src/hug_db.py
# HugDB
import faiss
import numpy as np
from cryptography.fernet import Fernet

class HugDB:

    def __init__(self, encryption_key):
        self.index = None
        self.id_to_data = {}
        self.encryption = Fernet(encryption_key)

    def encrypt_vector(self, vector):
        return self.encryption.encrypt(vector.tobytes())
    
    def decrypt_vector(self, encrypted_vector):
        return np.frombuffer(self.encryption.decrypt(encrypted_vector), dtype=np.float32)
    
    def build_index(self, data_vectors):
        data_vectors = np.array(data_vectors).astype('float32')
        self.index = faiss.IndexFlatL2(data_vectors.shape[1])
        self.index.add(data_vectors)

    def add_item(self, item_id, item_vector):
        item_vector = np.array(item_vector).astype('float32').reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatL2(item_vector.shape[1])
        faiss_id = self.index.ntotal
        self.index.add(item_vector)
        encrypted_vector = self.encrypt_vector(item_vector)
        self.id_to_data[item_id] = encrypted_vector
        print(f"Added item with id: {item_id}")
    
    def search(self, query_vector, k=5):
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        print(f"indices: {indices}")
        encrypted_item_vectors = [self.id_to_data[str(i)] for i in indices[0]]
        item_vectors = [self.decrypt_vector(encrypted_vector) for encrypted_vector in encrypted_item_vectors]
        return distances[0], item_vectors