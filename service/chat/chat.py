import os
from openai import OpenAI
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class chat:
    def __init__(self, model):
        self.client = None
        self.api_key = None
        self.load_env()
        self.model = model
        self.model_url_map = {
            "qwen_plus": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen_max": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "doubao-seed-1-8-251228": "https://ark.cn-beijing.volces.com/api/v3",
            "gpt-3.5-turbo": "https://api.bianxie.ai/v1",
            "gpt-4": "https://api.bianxie.ai/v1"
        }
        self.initialize()

    def load_env(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")

    def initialize(self):
        try:

            # 初始化OpenAI兼容接口的模型
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.model_url_map[self.model],
            )


        except Exception as e:
            logger.error(f"问答链初始化失败: {str(e)}")
            raise

    def chat(self, messages):
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return result
        except Exception as e:
            logger.error(f"发送失败: {str(e)}")
            raise
