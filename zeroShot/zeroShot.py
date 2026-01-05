import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class zeroShot:
    def get_prompt(self, question):
        try:
            custom_prompt = [
                {"role": "system",
                 "content": "This is an Architectural Decision Record for a software. Give a Decision corresponding to the Context provided by the User."},
                {"role": "user", "content":f"## Context \n{question}"}
            ]

            return custom_prompt

        except Exception as e:
            raise
