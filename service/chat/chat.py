import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_cache"
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

local_model_T5 = ["t5-small", "t5-base", "flan-t5-small", "flan-t5-base"]

class chat:
    def __init__(self, model):
        self.tokenizer = None
        self.client = None
        self.api_key = None
        self.load_env()
        self.model_name = model
        self.model=None
        self.model_url_map = {
            "qwen_plus": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen_max": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "doubao-seed-1-8-251228": "https://ark.cn-beijing.volces.com/api/v3",
            "gpt-3.5-turbo": "https://api.bianxie.ai/v1",
            "gpt-4": "https://api.bianxie.ai/v1",
            "gemini-1.5-flash": "https://api.bianxie.ai/v1",
            "claude-3-5-haiku-20241022": "https://api.bianxie.ai/v1",
            "gemini-1.5-pro": "https://api.bianxie.ai/v1",
        }
        self.initialize()

    def load_env(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")

    def initialize(self):
        try:
            if self.model_name in local_model_T5 :
                # 初始化本地 T5 模型
                temp_model_name = self.model_name
                if self.model_name in local_model_T5:
                    if self.model_name == "flan-t5-small" or self.model_name == "flan-t5-base":
                        temp_model_name = "google/"+self.model_name
                    self.tokenizer = T5Tokenizer.from_pretrained(temp_model_name)
                    self.model = T5ForConditionalGeneration.from_pretrained(temp_model_name)
                    logger.info(f"本地 T5 模型加载成功: {self.model_name}")
            elif self.model_name == "gpt2":
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
                logger.info(f"本地 GPT 模型加载成功: {self.model_name}")

            else:
                # 初始化OpenAI兼容接口的模型
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.model_url_map[self.model_name],
                )


        except Exception as e:
            logger.error(f"问答链初始化失败: {str(e)}")
            raise

    def chat(self, messages):
        try:
            if self.model_name in local_model_T5:  # 本地模型分支
                inputs = self.tokenizer(messages, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # 移动到 GPU/CPU

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=64,  # 调整为适合决策的长度
                        min_length=10,
                        num_beams=4,
                        early_stopping=True
                    )
                result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # 模拟 OpenAI 返回格式
                class MockChoice:
                    def __init__(self, content):
                        self.message = type('obj', (object,), {'content': content})

                class MockResult:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]

                return MockResult(result_text)
            elif self.model_name =="gpt2":
                inputs = self.tokenizer(messages, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # 移动到 GPU/CPU
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # 模拟 OpenAI 返回格式
                class MockChoice:
                    def __init__(self, content):
                        self.message = type('obj', (object,), {'content': content})

                class MockResult:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]

                return MockResult(result_text)
            else:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                return result
        except Exception as e:
            logger.error(f"发送失败: {str(e)}")
            raise
