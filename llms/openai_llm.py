"""
OpenAI LLM integration for the modular RAG framework.
Supports max_tokens, temperature, and top_p as configuration parameters for cost and quality control.
"""

import openai

from .base import LLM


class OpenAILLM(LLM):
    def __init__(self, model, api_key, max_tokens=512, temperature=0.7, top_p=1.0):
        """
        :param model: OpenAI chat model name
        :param api_key: OpenAI API key
        :param max_tokens: Maximum tokens for LLM response
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling parameter
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt):
        """
        Generate a response from the LLM given a prompt.
        :param prompt: Input prompt string
        :return: LLM response string
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
