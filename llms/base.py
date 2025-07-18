from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass
