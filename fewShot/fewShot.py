import os
from dotenv import load_dotenv
import logging

from service.PromptProvider import PromptProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class fewShot(PromptProvider):
    def __init__(self):
        load_dotenv()
        self.example_context1 = os.getenv("EXAMPLE_CONTEXT1")
        self.example_decision1 = os.getenv("EXAMPLE_DECISION1")
        self.example_context2 = os.getenv("EXAMPLE_CONTEXT2")
        self.example_decision2 = os.getenv("EXAMPLE_DECISION2")
    def get_prompt(self, question):
        try:
            custom_prompt = [
                {"role": "system",
                 "content": "These are architecture decision records. Follow the examples to get return Decision based on Context provided by the User."},
                {"role": "user", "content": f"## Context \n{self.example_context1}"},
                {"role": "assistant", "content": f"## Decision \n{self.example_decision1}"},
                {"role": "user", "content": f"## Context \n{self.example_context2}"},
                {"role": "assistant", "content": f"## Decision \n{self.example_decision2}"},
                {"role": "user", "content": f"## Context \n{question}"}
            ]

            return custom_prompt

        except Exception as e:
            raise
