import logging
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel

import service.retriever.retriever as rag
import service.chat.chat as ct
import zeroShot.zeroShot as zs
import fewShot.fewShot as fs

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str

class final:
    @app.post("/rag/")
    async def get_rag(request: QuestionRequest):
        try:
            question = request.question
            prompt = rag.retrieve().get_prompt(question)
            answer = ct.chat(model="qwen-plus").chat(prompt)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return {"error": f"抱歉，无法回答您的问题。错误信息: {str(e)}"}

    @app.post("/zeroshot/")
    async def get_zero_shot(request: QuestionRequest):
        try:
            question = request.question
            prompt = zs.zeroShot().get_prompt(question)
            answer = ct.chat(model="qwen-plus").chat(prompt)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return {"error": f"抱歉，无法回答您的问题。错误信息: {str(e)}"}

    @app.post("/fewshot/")
    async def get_few_shot(request: QuestionRequest):
        try:
            question = request.question
            prompt = fs.fewShot().get_prompt(question)
            answer = ct.chat(model="qwen-plus").chat(prompt)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return {"error": f"抱歉，无法回答您的问题。错误信息: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)