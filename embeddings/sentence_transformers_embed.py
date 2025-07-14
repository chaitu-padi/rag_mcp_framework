from sentence_transformers import SentenceTransformer
import torch

from .base import EmbeddingModel


class SentenceTransformersEmbedding(EmbeddingModel):
    def __init__(self, model="all-MiniLM-L6-v2", chunk_size=512):
        # Always use 'cuda' if available, else 'cpu'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Workaround: do not pass device to SentenceTransformer, move after loading
        self.model = SentenceTransformer(model)
        try:
            self.model = self.model.to(device)
        except NotImplementedError:
            # If NotImplementedError (meta tensor), use to_empty
            self.model = self.model.to_empty(device=device)
        self.chunk_size = chunk_size

    def embed(self, texts):
        # Optionally chunk texts if needed
        return self.model.encode(
            texts, batch_size=self.chunk_size, show_progress_bar=False
        ).tolist()
