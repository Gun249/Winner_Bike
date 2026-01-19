import os
import asyncio
import re
from lightrag import LightRAG, QueryParam
from sympy import re
from lib.logger import logger
from tqdm import tqdm
from fastapi import FastAPI , HTTPException
import uvicorn
from pydantic import BaseModel
from lib.initialize_lightrag import initialize_lightrag
from lib.pdf import load_pdfs_to_rag
from typing import List, Optional, Dict, Any
import traceback

import json
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = FastAPI(title="RAG System with LightRAG and Gemini-2.5 flash", version="1.0.0")

class QueryRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, Any]] = []
    mode: str = "global"  

rag = None

@app.on_event("startup")
async def startup_event():
    global rag
    logger.info(" Starting up and initializing LightRAG...")
    rag = await initialize_lightrag()
    await rag.initialize_storages()
    logger.info("System is ready to use.")

@app.on_event("shutdown")
async def shutdown_event():
    global rag
    if rag:
        logger.info(" Shutting down and finalizing storages...")
        await rag.finalize_storages()
        logger.info(" Shutdown complete.")  

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.message[:50]}...")
        

        formatted_history = ""
        if request.chat_history:
            formatted_history = "--- Conversation History ---\n"
            for msg in request.chat_history:

                role = msg.get("role", "Unknown")
                content = msg.get("content", "")
                

                display_role = "User" if role.lower() == "user" else "AI"
                
                formatted_history += f"{display_role}: {content}\n"
            formatted_history += "----------------------------\n"


        prompt = f"""
        {formatted_history}

        Current Question: {request.message}

        Instruction: You are a friendly and knowledgeable assistant. Use the provided context to answer the Current Question accurately.
        """

        # print(prompt) 


        result = await rag.aquery(
            prompt,
            param=QueryParam(mode=request.mode)
        )

        return {'response': result}

    except Exception as e:
        logger.error(f"Error processing query: {e}")

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_sale")
async def query_sale_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received sales query: {request.message[:50]}...")
        

        formatted_history = ""
        if request.chat_history:
            formatted_history = "--- Conversation History ---\n"
            for msg in request.chat_history:

                role = msg.get("role", "Unknown")
                content = msg.get("content", "")
                

                display_role = "User" if role.lower() == "user" else "AI"
                
                formatted_history += f"{display_role}: {content}\n"
            formatted_history += "----------------------------\n"


        prompt = f"""
        {formatted_history}

        Current Question: {request.message}

        Instruction: You are a friendly and enthusiastic sales representative. Use the provided context to answer the Current Question and aim to close the sale effectively.
        """

        # print(prompt) 


        result = await rag.aquery(
            prompt,
            param=QueryParam(mode=request.mode)
        )

        return {'response': result}

    except Exception as e:
        logger.error(f"Error processing sales query: {e}")

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)