import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.database.vectorstore import create_rag_chain

load_dotenv()

app = FastAPI(title="Medical ChatBot API")

class Query(BaseModel):
    question: str

# Initialize the RAG chain
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot-index")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

qa_chain = None
if PINECONE_API_KEY and HUGGINGFACEHUB_API_TOKEN:
    qa_chain = create_rag_chain(
        INDEX_NAME,
        PINECONE_API_KEY,
        "sentence-transformers/all-MiniLM-L6-v2",
        HUGGINGFACEHUB_API_TOKEN
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Medical ChatBot API"}

@app.post("/ask")
async def ask_question(query: Query):
    if not qa_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized. Check API keys.")

    try:
        response = qa_chain.run(query.question)
        return {"question": query.question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
