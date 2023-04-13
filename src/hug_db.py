# src/hug_db.py
# HugDB
import faiss
import numpy as np
from cryptography.fernet import Fernet

class HugItem:
    def __init__(self, item_id, metadata, encrypted_vector):
        self.item_id = item_id
        self.metadata = metadata
        self.encrypted_vector = encrypted_vector

class HugDB:
    def __init__(self, encryption_key=None, nlist=100, nprobe=10, num_shards=1, M=16, nbits=8, index_filename=None):
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.encryption = Fernet(encryption_key)
        self.nlist = nlist
        self.nprobe = nprobe
        self.num_shards = num_shards
        self.M = M
        self.nbits = nbits
        self.index_filename = index_filename
        self.items = {}
        self.index = None

    def encrypt_vector(self, vector):
        return self.encryption.encrypt(vector.tobytes())
    
    def decrypt_vector(self, encrypted_vector):
        return np.frombuffer(self.encryption.decrypt(encrypted_vector), dtype=np.float32)
    
    def build_index(self, data_vectors, metadata_list):
        data_vectors = np.array(data_vectors).astype('float32')
        quantizer = faiss.IndexFlatL2(data_vectors.shape[1])
        index = faiss.IndexIVFPQ(quantizer, data_vectors.shape[1], self.nlist, self.M, self.nbits)
        index.train(data_vectors)

        self.index = faiss.IndexShards(data_vectors.shape[1], self.num_shards)

        for i, (vector, metadata) in enumerate(zip(data_vectors, metadata_list)):
            self.add_item(str(i), vector, metadata)

        if self.index_filename is not None:
            faiss.write_index(self.index, self.index_filename)

    def add_item(self, item_id, item_vector, metadata):
        item_vector = np.array(item_vector).astype('float32').reshape(1, -1)
        encrypted_vector = self.encrypt_vector(item_vector)
        item = HugItem(item_id, metadata, encrypted_vector)
        self.items[item_id] = item
        self.index.add(item_vector)
        print(f"Added item with id: {item_id}") #used for testing. (Remove at a later pull)
    
    def search(self, query_vector, k=5, metadata_filter=None):
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        self.index.nprobe = self.nprobe # can also be adjustable 1 - 10
        distances, indices = self.index.search(query_vector, k)

        if metadata_filter is not None:
            filtered_indices, filtered_distances = [], []
            for i, dist in zip(indices[0], distances[0]):
                item = self.items[str(i)]
                if metadata_filter(item.metadata):
                    filtered_indices.append(i)
                    filtered_distances.append(dist)
            indices, distances = np.array(filtered_indices).reshape(1, -1), np.array(filtered_distances).reshape(1, -1)
        
        print(f"indices: {indices}") #used for testing. (Remove at a later pull)
        encrypted_item_vectors = [self.items[str(i)].encrypted_vector for i in indices[0]]
        item_vectors = [self.decrypt_vector(encrypted_vector) for encrypted_vector in encrypted_item_vectors]
        return distances[0], item_vectors