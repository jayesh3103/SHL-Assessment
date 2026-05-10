from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any
import uvicorn

from agent import handle_chat

app = FastAPI(title="SHL Conversational Assessment Recommender")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Pass messages to our agent logic
    # The assignment says the POST carries full stateless history
    # Example format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    response_data = handle_chat(request.messages)
    return response_data

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
