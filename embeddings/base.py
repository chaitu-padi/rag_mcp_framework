from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts):
        pass
