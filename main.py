# Standard library imports
import json
import os
import asyncio
import re
from typing import List, Optional, Dict, Any

# Third-party imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm

# LightRAG imports
from lightrag import QueryParam

# Local imports
from lib.logger import logger
from lib.initialize_lightrag import initialize_lightrag
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

def create_check_stock_logic(inventory_data: List[Dict[str, Any]]):
    """Factory function to create check_stock_logic with inventory data"""
    def check_stock_logic(model_name: str) -> str:
        """Check stock logic and return formatted string"""
        for item in inventory_data:
            if item.get("product_name", "").lower() == model_name.lower():
                stock = item.get("stock_quantity", 0)
                price = item.get("price", "N/A")

                if stock > 0:
                    return f"✅ Available | Model: {model_name} | Price: {price} บาท | Stock: {stock} units"
                else:
                    return f"❌ Out of Stock | Model: {model_name} | INSTRUCTION: Call lightrag_tool(query='{model_name} alternatives') immediately."
        return f"❓ Not Found | Model: {model_name} not in inventory."
    return check_stock_logic

# Tool schema for function calling
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "check_stock_logic",
            "description": "Check real-time stock and price. Use this whenever user asks about availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model name to check (e.g. 'PCX 160', 'Wave')."
                    }
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lightrag_tool",
            "description": "Retrieve technical specs, pros/cons, and FIND ALTERNATIVES if stock is empty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword (e.g. 'PCX 160 specs', 'PCX 160 competitors')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]



def check_tool_call_in_text(content: str, tool_calls: List) -> Optional[List]:
    """Check for tool call in text content and extract JSON arguments"""
    if not tool_calls and content and "<tool_call>" in content:
        logger.warning("⚠️ Detected text-based tool call")
        
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            try:
                tool_data = json.loads(json_str)
                
                class FakeToolCall:
                    def __init__(self, name, args):
                        self.id = "call_fake_123"
                        self.type = "function"
                        self.function = type('obj', (object,), {
                            'name': name,
                            'arguments': json.dumps(args)
                        })

                return [FakeToolCall(tool_data["name"], tool_data["arguments"])]
                
            except Exception as e:
                logger.error(f"❌ Error parsing tool call: {e}")
    return None


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

@app.post("/run_chat")
async def run_chat(query: RunChatRequest) -> Dict[str, str]:
    """Chat endpoint with tool calling support"""
    
    system_prompt_for_typhoon = """
        ### ROLE & PERSONA
        You are a motorcycle consultant at "Winner Bike" (Thai local shop style).
        - Speak Thai, friendly, honest. Use "ผม" and "ครับ".
        - Answer directly. Short and clear. No fluff.

        ### TOOLS (Use them, don't guess)
        1. `check_stock_logic`: Check availability/price
        2. `lightrag_tool`: Get specs/alternatives

        ### WORKFLOW
        **Stock Check:**
        - Call check_stock_logic first
        - ✅ Available → Say: "มีครับ [Model] ราคา [Price] บาท"
        - ❌ Out of Stock → Call lightrag_tool for alternatives, then recommend

        **Explain Product:**
        - Trigger: "สเปค", "ดียังไง", "อธิบายหน่อย"
        - Call lightrag_tool → Summarize in plain Thai (focus on benefits)

        **Recommendation:**
        - Trigger: "แนะนำรถ", "รถไหนดี"
        - Call lightrag_tool → Only recommend what the tool returns

        ### RESPONSE RULES (CRITICAL)
        Your response MUST be a polished Thai customer chat reply that:
        1. Answers ONLY what the customer asked - don't volunteer extra info
        2. Keeps it SHORT and DIRECT - no lengthy explanations
        3. Sounds like a real person chatting, NOT an advertisement or sales pitch
        4. Does NOT introduce new topics unless the customer asks
        5. Provides technical details ONLY when customer explicitly requests them
        6. Is conversational and natural (like talking to a friend at a shop)
        7. Never mentions "draft", "raw data", "database", or technical processes

        ### STRICT RULES
        - Always respond in FINAL Thai. No English unless technical terms.
        - Don't lie about stock availability.
        - Be honest if you don't know something.
    """
    
    logger.info(f"Running chat for query: {query.message[:50]}...")
    
    # Create closure with inventory data
    check_stock_fn = create_check_stock_logic(query.Data_model_stock_price)
    
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
                    function_response = await lightrag_tool(query=function_args.get("query"))
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