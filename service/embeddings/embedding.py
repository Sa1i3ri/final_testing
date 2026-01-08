import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEmbedding:
    def __init__(self):
        self.embeddings = None
        self.local_faiss_index_path = None
        self.local_data = None
        self.load_env()# 存 HuggingFace 句向量模型
        env_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")
        self.initialize_embeddings(env_model_path)

    def load_env(self):
        load_dotenv()
        self.local_data = os.getenv("LOCAL_DATA")
        self.local_faiss_index_path = os.getenv("LOCAL_FAISS_INDEX_PATH")


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

    def load_csv_documents(self, csv_file_path):
        """加载并处理CSV文档"""
        self.documents = []

        try:
            # 读取CSV内容
            import csv
            with open(csv_file_path, mode='r', encoding='latin1') as file:
                reader = list(csv.DictReader(file))  # 将数据加载为列表
                for row in reader[:-100]:  # 跳过最后100条数据
                    md_content = row.get("md_content", "").strip()

                    if md_content:
                        self.documents.append(md_content)


            logger.info(f"成功加载CSV文档: {csv_file_path}，共{len(self.documents)}个文档")

        except Exception as e:
            logger.error(f"加载CSV文档失败 {csv_file_path}: {str(e)}")

        return len(self.documents)

    def save_vector_store(self,vector_store, file_path):
        """将向量存储保存到磁盘"""
        if not vector_store:
            raise ValueError("向量存储未初始化，请先创建向量存储")

        try:
            vector_store.save_local(file_path)
            logger.info(f"向量存储已保存到磁盘路径: {file_path}")
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise

    def create_vector_store(self):
        try:
            index_path = self.local_faiss_index_path
            if os.path.exists(index_path):
                # 如果本地磁盘已有索引，则加载
                vector_store = FAISS.load_local(index_path, self.embeddings)
                logger.info(f"成功加载本地向量存储: {index_path}")
                return vector_store
            else:
                self.load_csv_documents(self.local_data)
                if not self.documents:
                    raise ValueError("没有可处理的文档，请先加载文档")

                # 创建新的向量存储
                vector_store = FAISS.from_texts(texts=self.documents, embedding=self.embeddings)
                self.save_vector_store(vector_store,index_path)
                logger.info(f"向量存储创建成功，包含 {len(self.documents)} 个文本块")
                return vector_store

        except Exception as e:
            logger.error(f"向量存储创建或加载失败: {str(e)}")
            raise