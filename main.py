import os
from openai import OpenAI
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
from dotenv import load_dotenv
import logging

from langchain_core.callbacks import CallbackManagerForChainRun

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGAssistant:
    def __init__(self, model_path=None):
        self.vector_store = None  # 后面存 FAISS 索引
        self.qa_chain = None  # 后面存 RetrievalQA 链条
        self.embeddings = None  # 后面存 HuggingFace 句向量模型
        self.documents = []
        self.load_env()
        self.client = None
        # 优先使用环境变量中的模型路径，如果没有则使用传入的model_path
        env_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")
        self.initialize_embeddings(env_model_path if env_model_path else model_path)

    def load_env(self):
        load_dotenv()
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.local_embedding_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")

        if not self.qwen_api_key:
            logger.warning("QWEN_API_KEY 环境变量未设置，将在Streamlit界面中提示用户输入")

        if self.local_embedding_model_path:
            logger.info(f"从环境变量获取到本地嵌入模型路径: {self.local_embedding_model_path}")
        else:
            logger.info("未在环境变量中找到本地嵌入模型路径(LOCAL_EMBEDDING_MODEL_PATH)")

    def initialize_embeddings(self, model_path=None):
        try:
            # 优先使用本地模型
            if model_path and os.path.exists(model_path):
                # 从本地文件加载模型
                self.embeddings = HuggingFaceEmbeddings(model_name=model_path)
                logger.info(f"成功从本地路径加载嵌入模型: {model_path}")
            else:
                # 检查是否提供了模型路径但路径不存在
                if model_path:
                    logger.error(f"本地模型路径不存在: {model_path}")
                    raise FileNotFoundError(f"本地模型路径不存在: {model_path}")

        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {str(e)}")
            raise

    def load_csv_documents(self, csv_files):
        """加载并处理CSV文档"""
        self.documents = []

        for csv_file in csv_files:
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(csv_file.read())
                    tmp_file_path = tmp_file.name

                # 读取CSV内容
                import csv
                with open(tmp_file_path, mode='r', encoding='latin1') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        star = row.get("stars", "").strip()
                        md_content = row.get("md_content", "").strip()

                        if md_content and star.isdigit() and int(star) > 1000:  # 只添加stars大于100且非空内容
                            self.documents.append(md_content)

                # 清理临时文件
                os.unlink(tmp_file_path)

                logger.info(f"成功加载CSV文档: {csv_file.name}，共{len(self.documents)}个文档")

            except Exception as e:
                logger.error(f"加载CSV文档失败 {csv_file.name}: {str(e)}")
                raise

        return len(self.documents)

    def create_vector_store(self):
        """创建向量存储"""
        try:
            if not self.documents:
                raise ValueError("没有可处理的文档，请先加载PDF文件")

            # 创建向量存储
            self.vector_store = FAISS.from_texts(texts=self.documents, embedding=self.embeddings)
            logger.info(f"向量存储创建成功，包含 {len(self.documents)} 个文本块")

        except Exception as e:
            logger.error(f"向量存储创建失败: {str(e)}")
            raise

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


# 创建Streamlit界面
def create_streamlit_app():
    st.set_page_config(
        page_title="RAG for ADR",
        layout="wide"
    )

    # 初始化RAG助手
    if "rag_assistant" not in st.session_state:
        try:
            st.session_state.rag_assistant = RAGAssistant()
            st.success("成功初始化RAG助手，使用本地嵌入模型")
        except Exception as e:
            st.error(f"初始化RAG助手失败: {str(e)}")
            return

    # 从环境变量加载默认配置
    default_api_key = os.getenv("QWEN_API_KEY", "")
    default_model_url = os.getenv("LLM_MODEL_URL", "")
    local_embedding_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "")

    st.title("RAG for ADR")

    # 显示当前使用的模型信息

    # 侧边栏 - 上传PDF文档和API Key设置
    with st.sidebar:
        st.header("设置")

        # API Key输入，默认值为从环境变量读取的值
        qwen_api_key = default_api_key

        # 添加模型URL输入，默认使用环境变量中的值
        qwen_model_url = default_model_url

        # PDF上传
        st.subheader("上传文档")
        pdf_files = st.file_uploader(
            "支持多文件上传",
            type="csv",
            accept_multiple_files=True,
        )

        # 处理按钮
        if st.button("处理文档"):
            # 如果界面中未输入API Key，但环境变量中有，则使用环境变量中的
            if not qwen_api_key and default_api_key:
                qwen_api_key = default_api_key

            if not qwen_api_key:
                st.error("请输入模型API Key")
            elif not pdf_files:
                st.error("请上传PDF文档")
            else:
                with st.spinner("正在处理文档，请稍候..."):
                    try:
                        # 加载文档
                        doc_count = st.session_state.rag_assistant.load_csv_documents(pdf_files)

                        # 创建向量存储
                        st.session_state.rag_assistant.create_vector_store()

                        # 初始化问答链，传递模型URL
                        st.session_state.rag_assistant.initialize_qa_chain(qwen_api_key, qwen_model_url)

                        st.success(f"成功处理了 {doc_count} 个文档，可以开始提问了！")
                        st.session_state.ready = True
                    except Exception as e:
                        st.error(f"处理文档失败: {str(e)}")

    # 主界面 - 提问区域
    st.subheader("请输入您的问题")


    # 问题输入框
    question = st.text_input(
        ""
    )

    # 提问按钮
    if st.button("获取答案"):
        if not question:
            st.error("请输入您的问题")
        else:
            with st.spinner("正在生成答案，请稍候..."):
                try:
                    answer,prompt = st.session_state.rag_assistant.get_answer(question)

                    # 显示答案
                    st.markdown("###  答案")
                    st.markdown(answer)
                    st.markdown(prompt)

                except Exception as e:
                    st.error(f"获取答案失败: {str(e)}")

    # 页脚信息
    st.markdown("\n")
    st.markdown("---")


if __name__ == "__main__":
    create_streamlit_app()