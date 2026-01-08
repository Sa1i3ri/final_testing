from abc import ABC, abstractmethod

class PromptProvider(ABC):
    @abstractmethod
    def get_prompt(self, question):
        pass