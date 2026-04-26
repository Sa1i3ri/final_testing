import logging

from service.PromptProvider import PromptProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class zeroShot(PromptProvider):
    def __init__( self, model_type):
        self.type = model_type
    def get_prompt(self, question):
        if self.type == 'GPT':
            custom_prompt = [
                {"role": "system",
                 "content": "This is an Architectural Decision Record for a software. Give a Decision corresponding to the Context provided by the User."},
                {"role": "user", "content": f"## Context \n{question}"}
            ]
            return custom_prompt

        else:
            custom_prompt = f"This is an Architectural Decision Record for a software. Give a Decision corresponding to the Context provided by the User.\n## Context:\n{question}\n"
            return custom_prompt

