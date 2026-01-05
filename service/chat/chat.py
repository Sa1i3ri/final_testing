import os
from openai import OpenAI
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class chat:
    def __init__(self):
        self.client = None
        self.api_key = None
        self.load_env()
        self.initialize()

    def load_env(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")

    def initialize(self):

        try:
            llm_params = {
                "api_key": self.api_key,
                "model": os.getenv("LLM_MODEL_NAME", "qwen-max")
            }

            env_model_url = os.getenv("LLM_MODEL_URL")
            if env_model_url:
                llm_params["base_url"] = env_model_url
                logger.info(f"使用环境变量中的模型URL: {env_model_url}")
            else:
                # 默认使用通义千问官方API地址
                llm_params["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                logger.info("使用默认的通义千问官方API地址")

            # 初始化OpenAI兼容接口的模型
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )


        except Exception as e:
            logger.error(f"问答链初始化失败: {str(e)}")
            raise

    def chat(self, messages):
        try:
            result = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages
            )
            return result
        except Exception as e:
            logger.error(f"发送失败: {str(e)}")
            raise
