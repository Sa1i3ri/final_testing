import os
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import logging
from langchain_core.callbacks import CallbackManagerForChainRun
import service.embeddings.embedding as em

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class retrieve:
    def __init__(self):
        self.qwen_api_key = None
        self.local_embedding_model_path = None
        self.vector_store = em.RAGEmbedding().create_vector_store()
        self.qa_chain = None  # 后面存 RetrievalQA 链条
        self.embeddings = None  # 后面存 HuggingFace 句向量模型
        self.documents = []
        self.load_env()
        self.initialize_qa_chain(api_key=self.qwen_api_key)

    def load_env(self):
        load_dotenv()
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.local_embedding_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")


    def initialize_qa_chain(self, api_key=None, model_url=None):
        """初始化问答链，支持离线LLM模型"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化，请先创建向量存储")


        try:
            llm_params = {
                "api_key": api_key,
                "model": os.getenv("LLM_MODEL_NAME", "qwen-max")
            }

            env_model_url = os.getenv("LLM_MODEL_URL")
            if model_url:
                llm_params["base_url"] = model_url
            elif env_model_url:
                llm_params["base_url"] = env_model_url
                logger.info(f"使用环境变量中的模型URL: {env_model_url}")
            else:
                # 默认使用通义千问官方API地址
                llm_params["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                logger.info("使用默认的通义千问官方API地址")

            # 初始化OpenAI兼容接口的模型
            llm = ChatOpenAI(**llm_params)
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.client = client
            # 创建检索问答链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )

            logger.info("问答链初始化成功")

        except Exception as e:
            logger.error(f"问答链初始化失败: {str(e)}")
            raise

    def split_adr_content(self,md_content):
        # Find the start of the "Context" and "Decision" sections
        context_start = md_content.find("Context")
        decision_start = md_content.find("Decision")

        # Extract the content for each section
        context = md_content[context_start:decision_start].strip()
        decision = md_content[decision_start:].strip()

        return context, decision

    def get_answer(self, question):
        """获取问题的答案"""
        if not self.qa_chain:
            raise ValueError("问答链未初始化，请先初始化问答链")

        try:
            # 提取 source_documents
            run_manager = CallbackManagerForChainRun.get_noop_manager()
            source_documents = self.qa_chain._get_docs(question, run_manager=run_manager)

            # 根据 source_documents 动态生成新的 prompt
            custom_prompt = [
                {"role": "system", "content": "These are architecture decision records.Follow the examples to get return Decision based on Context provided by the User"}
            ]

            for doc in source_documents:
                context_part, decision_part = self.split_adr_content(doc.page_content)
                custom_prompt.append({"role": "system", "content": f"## Context \n{context_part}"})
                custom_prompt.append({"role": "assistant", "content": f"## Decision \n{decision_part}"})
            custom_prompt.append({"role": "user", "content":f"## Context \n{question}"})

            logger.info(f"Prompt being sent to LLM: {custom_prompt}")
            result = self.client.chat.completions.create(
                model="qwen-plus",
                messages=custom_prompt
            )

            # 提取相关信息
            answer = result


            return answer,custom_prompt

        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return f"抱歉，无法回答您的问题。错误信息: {str(e)}"