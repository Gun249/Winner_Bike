from lightrag import QueryParam
from typing import List, Dict, Any , Optional
import re
import json
from .logger import logger
import asyncio


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
        for item in inventory_data:
            if item.get("product_name", "").lower() == model_name.lower():
                stock = item.get("stock_quantity", 0)
                price = item.get("price", "N/A")

                if stock > 0:
                    return f"✅ Available | Model: {model_name} | Price: {price} บาท | Stock: {stock} units"
                else:
                    return f"❌ Out of Stock | Model: {model_name} | INSTRUCTION: Call lightrag_tool(query='{model_name} alternatives') immediately."
            elif model_name.upper() == "ALL":
                inventory_list = []
                for inv_item in inventory_data:
                    inv_model = inv_item.get("product_name", "Unknown Model")
                    inv_price = inv_item.get("price", "N/A")
                    inv_stock = inv_item.get("stock_quantity", 0)
                    status = "มีของ" if inv_stock > 0 else "หมด"
                    inventory_list.append(f"- {inv_model}: {inv_price} บาท ({status})")
                return "ตอนนี้หน้าร้านมีตามนี้ครับ:\n" + "\n".join(inventory_list)
        return f"❓ Not Found | Model: {model_name} not in inventory."
    return check_stock_logic

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