import os
import asyncio
import re
from lightrag import LightRAG, QueryParam
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
    stock : int
    mode: str = "global"  


class expainn_product_Request(BaseModel):
    message: str
    chat_history: List[Dict[str, Any]] = []
    mode: str = "global"

rag = None

def clean_Reference(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'### References.*', '', text, flags=re.DOTALL)
    text = re.sub(r'อ้างอิง:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'แหล่งที่มา:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'URL:.*', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    return text.strip()

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


            Stock Availability: {request.stock}

            Instruction:
            If the requested product is in stock, clearly confirm availability.
            Keep the response short and direct.
            Do NOT explain specifications, features, or benefits unless the customer asks.
            You may ask one short, optional follow-up question.
            Wait for the customer to request more details before providing any technical explanation."""


        # print(prompt) 


        result = await rag.aquery(
            prompt,
            param=QueryParam(mode=request.mode)
        )

        return {'response': clean_Reference(result)}

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

        Stock Availability: {request.stock}

        Instruction:
        If the requested product is out of stock, inform the customer clearly and politely.
        Then suggest 1–2 closest available alternatives.
        Keep the response short and concise.
        Do NOT explain specifications, features, or comparisons.
        Wait for the customer to ask before providing more details.
        """

        # print(prompt) 


        result = await rag.aquery(
            prompt,
            param=QueryParam(mode=request.mode)
        )

        return {'response': clean_Reference(result)}
    except Exception as e:
        logger.error(f"Error processing sales query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_product")
async def explain_product_endpoint(request: expainn_product_Request):
    try:
        logger.info(f"Received product explanation query: {request.message[:50]}...")
        

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


        Instruction:
        Provide a clear and honest explanation of the product's specifications, features, and benefits.
        Keep the response informative yet concise.
        Do NOT introduce new topics on your own.
        Wait for the customer to ask before giving suggestions or alternatives.
        """

        # print(prompt) 


        result = await rag.aquery(
            prompt,
            param=QueryParam(mode=request.mode)
        )

        return {'response': clean_Reference(result)}


    except Exception as e:
        logger.error(f"Error processing sales query: {e}")

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)