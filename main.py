# Standard library imports
import json
import os
import asyncio
import re
from typing import List, Dict, Any

# Third-party imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI


# Local imports
from lib.logger import logger
from lib.initialize_lightrag import initialize_lightrag
from lib.tools import set_rag_instance, create_check_stock_logic,tools_schema, check_tool_call_in_text
# from lib.pdf import load_pdfs_to_rag


# ============================================
# Configuration & Initialization
# ============================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

client = OpenAI(
    api_key=os.getenv("TYPHOON_API_KEY"),
    base_url="https://api.opentyphoon.ai/v1"
)

app = FastAPI(
    title="RAG System with LightRAG and Gemini-2.5 flash",
    version="1.0.0"
)

# Global RAG instance
rag = None

# ============================================
# Class Definitions
# ============================================
class RunChatRequest(BaseModel):
    message: str
    Data_model_stock_price : List[Dict[str, Any]] = []
    chat_history: List[Dict[str, Any]] = []

# ============================================
# Helper Functions
# ============================================

def clean_Reference(text: str) -> str:
    """Remove references and citations from text"""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'### References.*', '', text, flags=re.DOTALL)
    text = re.sub(r'อ้างอิง:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'แหล่งที่มา:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'URL:.*', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    return text.strip()

def create_chat_history(chat):
        formatted_history = ""
        if chat:
            formatted_history = "--- Conversation History ---\n"
            for msg in chat:

                role = msg.get("role", "Unknown")
                content = msg.get("content", "")
                

                display_role = "User" if role.lower() == "user" else "AI"
                
                formatted_history += f"{display_role}: {content}\n"
            formatted_history += "----------------------------\n"
        return formatted_history


# ============================================
# Application Lifecycle Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag
    logger.info("Starting up and initializing LightRAG...")
    rag = await initialize_lightrag()
    await rag.initialize_storages()
    logger.info("System is ready to use.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag
    if rag:
        logger.info("Shutting down and finalizing storages...")
        await rag.finalize_storages()
        logger.info("Shutdown complete.")




# ============================================
# API Routes
# ============================================

@app.post("/run_chat")
async def run_chat(query: RunChatRequest) -> Dict[str, str]:
    """Chat endpoint with tool calling support"""
    
    system_prompt_for_typhoon = """
        ### ROLE & PERSONA
        You are a motorcycle consultant at "Winner Bike" (Thai local shop style).
        - Speak Thai, friendly, honest. Use "ผม" and "ครับ".
        - Answer directly. Short and clear. No fluff.

        ### TOOLS (Use them, don't guess)
        1. `check_stock_logic`: Check availability/price OR list all models.
        2. `lightrag_tool`: Get specs/alternatives.

        ### WORKFLOW (STRICT)

        1. **Specific Stock Check (ถามรุ่นเจาะจง)**
        - User: "มี PCX ไหม", "PCX ราคาเท่าไหร่"
        - Action: Call `check_stock_logic(model_name="PCX")`.
        - ✅ Available → "มีของครับ [Model] สี [Color] ราคา [Price] บาท"
        - ❌ Out of Stock → Call `lightrag_tool` for alternatives immediately.

        2. **General Inquiry (ถามว่ามีรถอะไรบ้าง)**
        - User: "ที่ร้านมีรถอะไรบ้าง", "มีรุ่นไหนแนะนำไหม", "ขอดูรายการรถหน่อย"
        - Action: Call `check_stock_logic(model_name="ALL")`.
        - Response: List the available models nicely (bullet points).
            "ตอนนี้หน้าร้านมีตามนี้ครับ:
            - [Model A]: [Price] บาท (มีของ)
            - [Model B]: [Price] บาท (หมด)"

        3. **Explain Product (ถามสเปค)**
        - Trigger: "สเปค", "ดียังไง", "อธิบายหน่อย"
        - Action: Call `lightrag_tool`. Summarize benefits.

        4. **Recommendation (ถามแนะนำรถ)**
        - Trigger: "ขับในเมืองรุ่นไหนดี", "รถออกทริป"
        - Action: Call `lightrag_tool`. Recommend ONLY based on tool results.

        ### RESPONSE RULES
        1. **Short & Direct:** Answer ONLY what is asked.
        2. **Natural Thai:** Chat like a friend ("พี่ลองดูตัวนี้ไหมครับ").
        3. **No Tech Jargon:** Never mention "database", "json", or "tool".
        4. **Honesty:** If stock is 0, say it's out of stock.
        """
    
    logger.info(f"Running chat for query: {query.message[:50]}...")
    
    # Create closure with inventory data
    check_stock_fn = create_check_stock_logic(query.Data_model_stock_price)
    rag_tool = await set_rag_instance(rag)
    
    chat_history = create_chat_history(query.chat_history)
    
    messages = [{"role": "system", "content": system_prompt_for_typhoon}]
    
    if chat_history:
        messages.append({"role": "user", "content": f"Chat History:\n{chat_history}"})
    
    messages.append({"role": "user", "content": query.message})
    
    MAX_LOOP = 3  
    count = 0
    
    while count < MAX_LOOP:
        count += 1
        logger.info(f"Loop {count}/{MAX_LOOP}")
        response = client.chat.completions.create(
            model="typhoon-v2.5-30b-a3b-instruct",
            messages=messages,
            temperature=0.2,
            max_completion_tokens=50000,  
            tools=tools_schema,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or check_tool_call_in_text(response_message.content, None)
        
        if tool_calls:
            logger.info(f"🔧 Tool calls: {len(tool_calls)}")
            messages.append(response_message)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                logger.info(f"→ {tool_name}: {function_args}")

                if tool_name == "lightrag_tool":
                    function_response = await rag_tool(query=function_args.get("query"))
                    logger.info(f"← lightrag: {function_response[:80]}...")
                elif tool_name == "check_stock_logic":
                    function_response = check_stock_fn(model_name=function_args.get("model_name"))
                    logger.info(f"← stock: {function_response}")
                else:
                    logger.warning(f"Unknown tool: {tool_name}")
                    continue
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": function_response
                })

            continue  
        else:
            logger.info("✓ Final response ready")
            return {'response': response_message.content}
    
    logger.warning(f"⚠️ Max loop ({MAX_LOOP}) reached")
    return {'response': response_message.content if response_message else "ขออภัยครับ เกิดข้อผิดพลาด"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)