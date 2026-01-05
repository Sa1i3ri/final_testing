import logging
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel

import service.retrieve.retrieve as rt

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str

class RAGAssistant:
    @app.post("/ask/")
    async def get_answer(request: QuestionRequest):
        question = request.question
        answer =  rt.retrieve().get_answer(question)
        return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)