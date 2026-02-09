
import json
import os
import asyncio
import re
from typing import List, Dict, Any


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI



from lib.logger import logger
from lib.initialize_lightrag import initialize_lightrag
from lib.tools import set_rag_instance, create_check_stock_logic,tools_schema, check_tool_call_in_text, web_search_tool
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
    You are a motorcycle consultant at "Winner Bike".
    - Address the customer as "คุณ". Use "ผม/ครับ".
    - Answer directly, short, and clear. No fluff.

    ### 🚨 STRICT KNOWLEDGE RULE (IMPORTANT)
    - DO NOT use your internal pre-trained knowledge about motorcycle specs or prices.
    - If a model is NOT in `check_stock_logic` AND NOT in `lightrag_tool`, you MUST admit you don't have the info and then call `web_search_tool`.
    - NEVER guess specs. If the tool response is empty or "I don't know", you MUST use the next tool in priority.

    ### TOOLS PRIORITY & FALLBACK
    1. `check_stock_logic`: Check this FIRST for any model mentioned.
    2. `lightrag_tool`: Check this SECOND for specs/details.
    3. `web_search_tool`: **MANDATORY FALLBACK.** Use this if:
       - The user asks about a model that is "Not Found" in our stock.
       - You need to compare two models but `lightrag_tool` only provides info for one of them.
       - The information from `lightrag_tool` is insufficient to answer the specific question.

    ### WORKFLOW (STRICT)

    1. **Greeting:** Greet "คุณ" warmly. No tools needed.
    2. **Stock & Specs Check:**
       - Call `check_stock_logic`. 
       - If model not in stock, call `lightrag_tool`.
       - **[CRITICAL]** If comparing A and B, and you only have info for A: 
         -> Call `web_search_tool` for B immediately. DO NOT answer using your own memory.
    
    3. **Comparison Logic:**
       - When comparing, if any data point (like price or engine spec) is missing from our internal tools, search it on the web.
       - Summarize the web data clearly but tell the user: "ข้อมูลนี้เป็นข้อมูลทั่วไปจากอินเทอร์เน็ตนะครับ"

    ### RESPONSE RULES
    1. **STRICTLY NO BOLD TEXT:** Do NOT use **PCX** or **Aerox**. Write as normal text.
    2. **Honesty:** If a model is not sold at our shop, say "ทางร้านไม่ได้จำหน่ายรุ่น [Model] ครับ" before giving other info.
    3. **No Tech Jargon:** No "database", "JSON", "Tool".
    4. **Pivot:** After giving info from Web Search, always recommend a similar model that we HAVE in stock.
    """
    
    logger.info(f"Running chat for query: {query.message[:50]}...")
    

    check_stock_fn = create_check_stock_logic(query.Data_model_stock_price)
    rag_tool = await set_rag_instance(rag)
    
    chat_history = create_chat_history(query.chat_history)

    logger.info(f"Chat history:\n{chat_history}")
    
    
    messages = [{"role": "system", "content": system_prompt_for_typhoon}]
    
    if chat_history:
        messages.append({"role": "user", "content": f"Chat History:\n{chat_history}"})
    
    messages.append({"role": "user", "content": query.message})
    
    MAX_LOOP = 7
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
            messages.append(response_message)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                logger.info(f"→ {tool_name}: {function_args}")

                if tool_name == "lightrag_tool":
                    function_response = await rag_tool(query=function_args.get("query"))
                    if function_response:
                        logger.info(f"lightrag: {function_response[:80]}...")
                    else:
                        logger.warning("lightrag: ได้รับคำตอบเป็นค่าว่าง (None)")
                elif tool_name == "check_stock_logic":
                    function_response = check_stock_fn(model_name=function_args.get("model_name"))
                    logger.info(f"stock: {function_response}")
                elif tool_name == "web_search_tool":
                    function_response = web_search_tool(query=function_args.get("query"), max_results=function_args.get("max_results",3))
                    logger.info(f"web_search: {function_response[:80]}...")
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
            logger.info("Final response ready")
            return {'response': response_message.content}
    
    logger.warning(f"⚠️ Max loop ({MAX_LOOP}) reached")
    return {'response': response_message.content if response_message else "ขออภัยครับ เกิดข้อผิดพลาด"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)