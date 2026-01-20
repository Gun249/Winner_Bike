# Standard library imports
import json
import os
import asyncio
import re
import traceback
from typing import List, Optional, Dict, Any

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm

# LightRAG imports
from lightrag import LightRAG, QueryParam

# Local imports
from lib.logger import logger
from lib.initialize_lightrag import initialize_lightrag
from lib.pdf import load_pdfs_to_rag


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
# Pydantic Models
# ============================================

class QueryRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, Any]] = []
    stock: int
    mode: str = "global"

class ExplainProductRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, Any]] = []
    mode: str = "global"

class RunChatRequest(BaseModel):
    message: str

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
# Tool Functions
# ============================================

async def lightrag_tool(query: str) -> str:
    """Query the LightRAG knowledge base"""
    result = await rag.aquery(
        query,
        param=QueryParam(mode="global")
    )
    return result

# Tool schema for function calling
tool_schema = [
    {
        "type": "function",
        "function": {
            "name": "lightrag_tool",
            "description": (
                "THE KNOWLEDGE BASE. Strictly use this tool to retrieve technical facts, "
                "specifications, compatibility, model comparisons, and troubleshooting advice. "
                "AI MUST use this tool before answering any technical questions to ensure accuracy. "
                "Do NOT use this for checking stock availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Targeted search keywords extracted from the user's request. "
                            "Example: If user asks 'Does PCX 160 use synthetic oil?', "
                            "the query should be 'PCX 160 synthetic oil recommendation'."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]



def check_tool_call_in_text(content: str,tool_calls :List) -> Optional[Dict[str, Any]]:
    """Check for tool call in text content and extract JSON arguments"""
    if not tool_calls and content and "<tool_call>" in content:
        print("⚠️ ตรวจพบ Tool Call ในรูปแบบ Text! กำลังแปลงข้อมูล...")
        
        # ใช้ Regex ดึง JSON ออกมาจาก Tag
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            try:
                tool_data = json.loads(json_str)
                
                # สร้าง Object หลอกๆ ขึ้นมาให้เหมือน OpenAI tool_calls
                class FakeToolCall:
                    def __init__(self, name, args):
                        self.id = "call_fake_123" # มั่ว ID ไป
                        self.type = "function"
                        self.function = type('obj', (object,), {'name': name, 'arguments': json.dumps(args)})

                # ยัดใส่ตัวแปร tool_calls เพื่อให้ Code ด้านล่างทำงานต่อได้
                tool_calls = [FakeToolCall(tool_data["name"], tool_data["arguments"])]

                return tool_calls
                
            except Exception as e:
                print(f"❌ Error parsing manual tool call: {e}")

# ============================================
# API Routes
# ============================================

@app.post("/run_chat")
async def run_chat(query: RunChatRequest) -> str:
    """Chat endpoint with tool calling support"""
    system_prompt_for_typhoon = """
    ### ROLE & PERSONA
    You are the "Technical Motorcycle & Parts Consultant & Seller" at "Winner Bike".
    - Your vibe: Professional mechanic-turned-seller. Experienced, honest, friendly (trusted local shop style).
    - Your goal: Help customers find the right bike/parts from Winner Bike using accurate facts.
    - You act like a real staff at Winner Bike replying in chat.

    ### TOOL USAGE PROTOCOL (CRITICAL)
    Winner Bike has a massive database. You MUST use `lightrag_tool` to get accurate facts.
    - **NEVER guess specs.** If asked about details/pros/cons, use the tool.
    - **Query Style:** Extract keywords (e.g., "PCX 160 pros cons", "best city riding motorcycle recommendations").

    ### SCENARIO LOGIC & WORKFLOW (STRICT)
    Analyze the user's intent and follow these cases:

    1. **CASE: CHECK STOCK (ถามสต็อก/มีของไหม)**
    - **Context:** You receive stock status (Available/Out of Stock).
    - **IF IN STOCK:** - Say: "มีของครับพี่ รุ่นนี้ที่ Winner Bike ยังมีสต็อกครับ"
        - Action: **STOP.** Do not explain specs yet. Wait for the user to ask.
    - **IF OUT OF STOCK:**
        - Say: "ขออภัยครับ ตัวนี้ที่ร้านหมดชั่วคราวครับ"
        - Action: **IMMEDIATELY call `lightrag_tool`** to find a substitute.
        - Response: "...แต่ผมแนะนำลองดูเป็น [Alternative Model] ไหมครับ สเปคใกล้กันมาก พี่สนใจให้ผมเล่าตัวนี้แทนไหม?"

    2. **CASE: EXPLAIN PRODUCT (ถามรายละเอียด/สเปค/ดียังไง)**
    - **Trigger:** User asks "ขอสเปค...", "รุ่นนี้ดียังไง", "ช่วยอธิบายหน่อย", "ช่วยอธิบายสินค้าหน่อย", or says "เล่ามาเลย".
    - **Action:** **IMMEDIATELY call `lightrag_tool`** with query "[Model Name] specs features selling points".
    - **Response:** Summarize the tool data focusing on real-world benefits (e.g., "ประหยัดน้ำมัน", "U-Box ใหญ่").

    3. **CASE: RECOMMENDATION (แนะนำรถตามการใช้งาน)**
    - **Trigger:** User asks "ขับในเมืองรุ่นไหนดี?", "แนะนำรถออกทริปหน่อย", "รถผู้หญิงขับง่ายๆ", or describe their usage style.
    - **Action:** **IMMEDIATELY call `lightrag_tool`** with query based on usage (e.g., "best motorcycles for city traffic", "touring motorcycle recommendations").
    - **Constraint:** **DO NOT** recommend any model based on your own knowledge. **ONLY** recommend models returned by the tool.
    - **Response:** "ถ้าเน้นใช้งานแบบ [User's Style] ผมแนะนำเป็นตัว [Model from Tool] ครับ เพราะ [Reason from Tool]"

    ### CONVERSATION RULES
    1. **Winner Bike Identity:** Refer to yourself as "ผม" or "ทางร้าน Winner Bike".
    2. **Short & Sharp:** Answer only what is asked. Don't overwhelm the customer.

    ### TONE & LANGUAGE
    - Language: Thai (ภาษาไทย).
    - Tone: Natural, friendly, mechanic style. ❌ No emojis. ❌ No formal "ท่าน/เรียนแจ้ง".
    - ❌ NO sales hype. Be realistic.

    ### STRICT RESTRICTIONS
    - Do not mention "database". Say "เดี๋ยวผมดูข้อมูลสเปคให้ครับ".
    - **Strictly recommend ONLY products found in the `lightrag_tool` results.**
    """
    logger.info(f"Running chat for query: {query.message[:50]}...")
    
    messages = [
        {"role": "system", "content": system_prompt_for_typhoon},
        {"role": "user", "content": query.message}
    ]

    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=messages,
        temperature=0.2,
        max_completion_tokens=50000,
        tools=tool_schema,
        tool_choice="auto"
    )
    logger.info("Processing tool calls if any...")
    response_message = response.choices[0].message
    logger.info(f"Response Message: {response_message}")
    
    tool_calls = response_message.tool_calls
    logger.info(f"Tool calls: {tool_calls}")

    tool_calls = check_tool_call_in_text(response_message.content, tool_calls)
    
    if tool_calls:
        logger.info(f"Tool calls detected: {len(tool_calls)}")
        messages.append(response_message)

        for tool_call in tool_calls:
            logger.info(f"Processing tool call: {tool_call.function.name}")
            tool_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if tool_name == "lightrag_tool":
                logger.info(f"Calling lightrag_tool with args: {function_args}")
                function_response = await lightrag_tool(query=function_args.get("query"))
                logger.info(f"Tool response: {function_response[:100]}...")
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": function_response
                })
        
        logger.info("Generating final response after tool calls...")
        
        refine_instruction = f"""
            Below is accurate raw information (Draft):
            "{response_message.content}"

            Task:
            Rewrite the draft into a Thai customer chat response.

            Rules:
            - Answer only what the customer asked.
            - Keep the response short and direct.
            - Do not add explanations unless required to answer the question.
            - Do not sound like an advertisement.
            - Do not introduce new topics on your own.
            - Provide deeper technical details only if the customer asks a follow-up question.
        """
        
        messages.append({"role": "user", "content": refine_instruction})
        
        final_response = client.chat.completions.create(
            model="typhoon-v2.5-30b-a3b-instruct",
            messages=messages,
            temperature=0.2,
            max_completion_tokens=50000
        )
        
        logger.info("Final response generated.")
        return final_response.choices[0].message.content
    else:
        logger.info("No tool calls detected.")
        return response_message.content

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query endpoint for in-stock products"""
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
    """Query endpoint for out-of-stock products with alternatives"""
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
async def explain_product_endpoint(request: ExplainProductRequest):
    """Endpoint for detailed product explanations"""
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

# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)