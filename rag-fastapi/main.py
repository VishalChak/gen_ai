from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline


## Define input schema

class QueryRequest(BaseModel):
    question:str

## Initilize FastAPI app

app = FastAPI()

llm_model = "compound-beta"
embed_model = "local:BAAI/bge-small-en-v1.5"
groq_api_key = ""
data_dir = "data"

rag = RAGPipeline(model = llm_model, api_key = groq_api_key, embed_model =embed_model , data_dir = data_dir)

## define endpoint

@app.post("/query")
def ask_question(request:QueryRequest):
    try:
        answer = rag.response(request.question)
        return {"answer" : answer}
    except Exception as e:
        raise HTTPException (status_code=500, detail=str(e))