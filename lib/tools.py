import httpx
from typing import List, Optional
import re
import json
from .logger import logger
import asyncio
from tavily import TavilyClient
from supabase import create_client, Client
import os

tavily = TavilyClient(api_key=os.getenv("tavily_KEY"))

# Supabase singleton client
supabase_client: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

async def lightrag_tool(query: str) -> str:
    api_url = os.getenv("LIGHTRAG_API_URL")
    
    if not api_url:
        logger.error("LIGHTRAG_API_URL is not set in environment variables")
        return "Error: LightRAG API URL not configured"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json={"query": query, "mode": "global"})
            response.raise_for_status()
            
            return response.json().get("response", "No answer found")
    except httpx.HTTPStatusError as e:
        logger.error(f"LightRAG HTTP error: {e.response.status_code} - {e.response.text}")
        return "Error querying LightRAG"
    except httpx.ConnectError:
        logger.error(f"LightRAG: Cannot connect to {api_url} - Is the server running?")
        return "Error: LightRAG server is not reachable"
    except Exception as e:
        logger.error(f"Exception during LightRAG API call: {type(e).__name__}: {e}")  # เพิ่ม type(e).__name__
        return "Exception occurred while querying LightRAG"


async def check_stock_logic(model_name: str) -> str:
    """Check stock from Supabase products table and return formatted string"""
    try:
        # 1. กรณีค้นหาทั้งหมด (ALL)
        if model_name.upper() == "ALL":
            # ยิง 2 query พร้อมกัน: DB กรองฝั่ง server แทน Python
            in_stock_res, out_stock_res = await asyncio.gather(
                asyncio.to_thread(
                    lambda: supabase_client.table("products")
                        .select("product_name, price, stock_quantity")
                        .gt("stock_quantity", 0)
                        .execute()
                ),
                asyncio.to_thread(
                    lambda: supabase_client.table("products")
                        .select("product_name")
                        .eq("stock_quantity", 0)
                        .execute()
                )
            )

            return json.dumps({
                "in_stock": in_stock_res.data,
                "out_of_stock": out_stock_res.data
            }, ensure_ascii=False)

        # 2. กรณีค้นหารายรุ่น — ใช้ ilike เพื่อ partial match (case-insensitive)
        search_query = model_name.strip()
        response = supabase_client.table("products").select("product_name, price, stock_quantity").ilike("product_name", f"%{search_query}%").execute()

        if response.data:
            return json.dumps(response.data[0], ensure_ascii=False)

        # 3. หาไม่เจอจริงๆ
        return json.dumps({"status": "not_found", "model": model_name}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Supabase check_stock error: {type(e).__name__}: {e}")
        return f"Error querying stock: {e}"

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
        
        pattern = r'<tool_call>(.*?)ground'
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