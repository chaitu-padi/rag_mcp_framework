from abc import ABC, abstractmethod


class VectorDB(ABC):
    @abstractmethod
    def add(self, embeddings, documents):
        pass

    @abstractmethod
    def query(self, embedding, top_k=5):
        pass
