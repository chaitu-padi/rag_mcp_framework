"""
OpenAI Embedding integration for the modular RAG framework.
Supports chunk_size and rank as configuration parameters for future extensibility.
"""

import openai

from .base import EmbeddingModel


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, model, api_key, chunk_size=512, rank=1):
        """
        :param model: OpenAI embedding model name
        :param api_key: OpenAI API key
        :param chunk_size: (optional) chunk size for batching or splitting (not used by default)
        :param rank: (optional) rank parameter for future use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.chunk_size = chunk_size
        self.rank = rank

    def embed(self, texts):
        """
        Generate embeddings for a list of texts using OpenAI API.
        :param texts: List of input strings
        :return: List of embedding vectors
        """
        # Optionally use chunk_size and rank in your logic if needed
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]
