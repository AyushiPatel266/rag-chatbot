from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import get_answer

app = FastAPI(title="RAG Chatbot")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question can't be empty")

    result = get_answer(request.question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])