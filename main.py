
import json
import os
import asyncio
import re
import uuid
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

from lib.logger import logger, request_id_ctx
from lib.tools import lightrag_tool, check_stock_logic, tools_schema, check_tool_call_in_text, web_search_tool



# ============================================
# Configuration & Initialization
# ============================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

app = FastAPI(
    title="RAG System with LightRAG and Gemini 2.5 Flash",
    version="1.0.0"
)


# ── Request-ID & Logging Middleware ──────────────────────────
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request_id_ctx.set(rid)

    start = time.perf_counter()
    logger.info(f"{request.method} {request.url.path} started")

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception during request")
        response = JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.0f}ms)"
    )
    response.headers["X-Request-ID"] = rid
    return response


# ── Lifecycle events ─────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("Winner Bike API started")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Winner Bike API shutting down")

# ============================================
# Class Definitions
# ============================================
class RunChatRequest(BaseModel):
    message: str
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
# API Routes
# ============================================

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/run_chat")
async def run_chat(query: RunChatRequest) -> Dict[str, str]:
    """Chat endpoint with tool calling support"""
    
    system_prompt = """
    ### ROLE & PERSONA
    You are a proactive motorcycle consultant at "Winner Bike".
    - You are helpful, polite, and eager to close deals.
    - **Goal:** Sell OUR inventory and **UPSELL** to higher-value models when appropriate.
    - Address the user as "คุณลูกค้า".
    - **Keep responses short, concise, and to the point.**

    ### 🚨 CRITICAL DATA RULES
    1. **SOURCE OF TRUTH:** Your internal knowledge is **INVALID**. You must **ONLY** rely on tools.
    2. **NO GUESSING:** If tools fail, say you don't have the info. **DO NOT INVENT** specs/parts.
    3. **STRICTLY NO MARKDOWN:** Do NOT use `**` or `***`. Write plain text only.

    ### TOOLS STRATEGY
    1. **`check_stock_logic`:** Check availability (Inventory). Returns JSON with fields: product_name, price, stock_quantity.
       - If `stock_quantity > 0`: item is available.
       - If `stock_quantity == 0`: item is OUT OF STOCK → immediately call `lightrag_tool(query='<model_name> alternatives')` to find alternatives.
       - If `status == "not_found"`: model not in inventory → call `lightrag_tool` or `web_search_tool` for info.
    2. **`lightrag_tool`:** Internal Database (Specs, Parts, Upsell candidates).
    3. **`web_search_tool`:** External Web (Fallback).
    4. **CASUAL TALK:** No tools for greetings/thanks.

    ### 📈 UPSELL STRATEGY (The Art of Selling)
    When the user is interested in a specific model or category, ALWAYS try to suggest a **better/higher-spec model** that is **IN STOCK**.
    1. **Identify the Upgrade:**
       - If user looks for 110-125cc (e.g., Wave, Scoopy) -> Upsell to 160cc (e.g., PCX, Click 160, Lead).
       - If user looks for Standard -> Upsell to ABS / Hybrid / Keyless versions.
    2. **Check Stock of Upgrade:** Use `check_stock_logic` for the upsell model.
    3. **The Pitch:**
       - If Upgrade is Available: "รุ่น [A] ดีครับ แต่ถ้าคุณลูกค้าสนใจสมรรถนะที่สูงขึ้น ผมขอแนะนำ [B] ที่เรามีของพร้อมรับเลยครับ ตัวนี้จะได้ [Feature เด่น] เพิ่มเข้ามาครับ"
       - If Upgrade is Out of Stock: Just sell the original request.

    ### 🔍 UNIVERSAL SEARCH FLOW (Specs & Parts)
    Apply this when asking about Specs or Parts:
    1. **Internal Search:** Call `lightrag_tool` (e.g., "Wave 110i specs").
    2. **External Fallback:** If not found, use `web_search_tool` with Model Name included.
    3. **Answer:** Summarize based on tool results only.

    ### 🏍️ RECOMMENDATION FLOW (With Upsell)
    If user asks for a recommendation (e.g., "City riding"):
    1. **Search:** Use `lightrag_tool` to find candidates (Low & High tier).
    2. **Verify Stock:** Use `check_stock_logic` for ALL.
    3. **Filter:** Discard out-of-stock items silently.
    4. **Final Output:** Present the requested option AND try to Upsell.
       - *Example:* "สำหรับการขับในเมือง แนะนำ Grand Filano (มีของ) ครับ แต่ถ้าชอบความแรงและที่เก็บของกว้างกว่า ผมเชียร์ Lead 125 (มีของ) เพิ่มอีกรุ่นครับ"

    ### RESPONSE RULES
    1. **Format:** Plain text only. NO `**`, `***`.
    2. **Conciseness:** Answer directly.
    3. **Sales Mindset:** Always check if there is a more expensive/better bike in stock to mention before finishing the response.
    """
    
    logger.info(f"Running chat for query: {query.message[:50]}...")
    

    chat_history = create_chat_history(query.chat_history)

    
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_history:
        messages.append({"role": "user", "content": f"Chat History:\n{chat_history}"})
    
    messages.append({"role": "user", "content": query.message})
    
    MAX_LOOP = 5
    count = 0
    
    while count < MAX_LOOP:
        count += 1
        logger.info(f"Loop {count}/{MAX_LOOP}")
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            temperature=0.2,
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
                    function_response = await lightrag_tool(query=function_args.get("query"))
                    if function_response:
                        logger.info(f"lightrag: {function_response[:80]}...")
                    else:
                        logger.warning("lightrag: ได้รับคำตอบเป็นค่าว่าง (None)")
                elif tool_name == "check_stock_logic":
                    function_response = await check_stock_logic(model_name=function_args.get("model_name"))
                    logger.info(f"stock: {function_response}")
                elif tool_name == "web_search_tool":
                    function_response = web_search_tool(query=function_args.get("query"))
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)