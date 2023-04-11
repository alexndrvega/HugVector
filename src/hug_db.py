# HugVector
import faiss

class HugDB:

    def __init__(self):
        self.index = None
    
    def build_index(self, vectors):
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
    
    def search(self, query_vector, k=10):
        distances, indices = self.index.search(query_vector, k)
        return indices