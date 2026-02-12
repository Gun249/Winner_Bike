from lightrag import QueryParam
from typing import List, Dict, Any , Optional
import re
import json
from .logger import logger
import asyncio
from tavily import TavilyClient
import os

tavily = TavilyClient(api_key=os.getenv("tavily_KEY"))

async def set_rag_instance(rag_instance):
    rag = rag_instance
    async def lightrag_tool(query: str) -> str:
        """Query the LightRAG knowledge base"""
        result = await rag.aquery(
            query,
            param=QueryParam(mode="global")
        )
        return result
    return lightrag_tool

def create_check_stock_logic(inventory_data: List[Dict[str, Any]]):
    """Factory function to create check_stock_logic with inventory data"""
    def check_stock_logic(model_name: str) -> str:
        """Check stock logic and return formatted string"""
        
        # 1. กรณีค้นหาทั้งหมด (ALL)
        if model_name.upper() == "ALL":
            inventory_list = []
            for inv_item in inventory_data:
                inv_model = inv_item.get("product_name", "Unknown Model")
                inv_price = inv_item.get("price", "N/A")
                inv_stock = inv_item.get("stock_quantity", 0)
                status = "มีของ" if inv_stock > 0 else "หมด"
                inventory_list.append(f"- {inv_model}: {inv_price} บาท ({status})")
            return "ตอนนี้หน้าร้านมีตามนี้ครับ:\n" + "\n".join(inventory_list)

        # 2. กรณีค้นหารายรุ่น (Partial Match Logic)
        # ตัดช่องว่างหัวท้ายและแปลงเป็นตัวพิมพ์เล็ก
        search_query = model_name.lower().strip()
        
        for item in inventory_data:
            db_product_name = item.get("product_name", "").lower()
            
            # --- แก้ไขจุดนี้: ใช้ 'in' แทน '==' ---
            # เพื่อให้ "Fazzio" ค้นเจอใน "Yamaha Fazzio Hybrid" ได้
            if search_query in db_product_name:
                stock = item.get("stock_quantity", 0)
                price = item.get("price", "N/A")
                full_name = item.get("product_name") # ดึงชื่อเต็มจาก DB มาแสดง

                if stock > 0:
                    return f"✅ Available | Model: {full_name} | Price: {price} บาท | Stock: {stock} units"
                else:
                    return f"❌ Out of Stock | Model: {full_name} | INSTRUCTION: Call lightrag_tool(query='{model_name} alternatives') immediately."

        # 3. หาไม่เจอจริงๆ
        return f"❓ Not Found | Model: {model_name} not in inventory."
        
    return check_stock_logic

def web_search_tool(query: str) -> str:
    logger.info(f"Performing web search for query: {query}")
    try:
        response = tavily.search(query=query, max_results=3,search_depth="advanced")

        context = "ผลลัพธ์การค้นหาเว็บ:\n"
        for result in response['results']:
            context += f"- {result['title']}: {result['content']}\n"
        return context
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return "❌ เกิดข้อผิดพลาดในการค้นหาเว็บ"
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "check_stock_logic",
            "description": "Check real-time stock and price. Use this for specific models OR to list all available inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The specific model name (e.g. 'PCX 160'). IF user asks 'what models do you have?' or 'list all', pass the string 'ALL'."
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
            "description": "Retrieve technical specs, pros/cons, and FIND ALTERNATIVES if stock is empty. ",
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
    },
    {
    "type": "function",
    "function": {
        "name": "web_search_tool",
        "description": "FALLBACK TOOL: Use this ONLY when internal tools (check_stock/lightrag) return no results. specific use cases: 1. Finding technical specifications (engine, cc, weight) for models not in our inventory. 2. Looking up competitor models for comparison. 3. Checking general market prices or launch dates in Thailand.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search keywords. Combine model name with context terms like 'specs', 'price thailand', 'review', or 'comparison'. Example: 'Yamaha XMAX 2024 specs thailand'"
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
        logger.warning("Detected text-based tool call")
        
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            try:
                tool_data = json.loads(json_str)
                
                #Mock Data for OpenAI tool call structure
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
                logger.error(f"Error parsing tool call: {e}")
    return None