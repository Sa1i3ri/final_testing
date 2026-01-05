from dotenv import load_dotenv
import logging
import service.embeddings.embedding as em

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class retrieve:
    def __init__(self):
        self.vector_store = em.RAGEmbedding().create_vector_store()
        self.load_env()

    def load_env(self):
        load_dotenv()


    def split_adr_content(self,md_content):
        # Find the start of the "Context" and "Decision" sections
        context_start = md_content.find("Context")
        decision_start = md_content.find("Decision")

        # Extract the content for each section
        context = md_content[context_start:decision_start].strip()
        decision = md_content[decision_start:].strip()

        return context, decision

    def get_prompt(self, question):
        try:
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            source_documents = retriever.get_relevant_documents(question)

            # 根据 source_documents 动态生成新的 prompt
            custom_prompt = [
                {"role": "system",
                 "content": "These are architecture decision records.Follow the examples to get return Decision based on Context provided by the User"}
            ]

            for doc in source_documents:
                context_part, decision_part = self.split_adr_content(doc.page_content)
                custom_prompt.append({"role": "system", "content": f"## Context \n{context_part}"})
                custom_prompt.append({"role": "assistant", "content": f"## Decision \n{decision_part}"})
            custom_prompt.append({"role": "user", "content": f"## Context \n{question}"})

            logger.info(f"Prompt being sent to LLM: {custom_prompt}")

            return custom_prompt

        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return f"抱歉，无法回答您的问题。错误信息: {str(e)}"
