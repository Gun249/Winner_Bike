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
        async with httpx.AsyncClient(timeout=60.0) as http:
            response = await http.post(api_url, json={"query": query, "mode": "global"})
            response.raise_for_status()
            result = response.json().get("response", "No answer found")
            logger.debug(f"LightRAG response length: {len(result)} chars")
            return result
    except httpx.HTTPStatusError as e:
        logger.error(
            f"LightRAG HTTP error: {e.response.status_code}",
            exc_info=True,
        )
        return "Error querying LightRAG"
    except httpx.ConnectError:
        logger.error("LightRAG: Cannot connect to server – is it running?")
        return "Error: LightRAG server is not reachable"
    except Exception:
        logger.exception("Unhandled exception during LightRAG API call")
        return "Exception occurred while querying LightRAG"


async def check_stock_logic(model_name: str) -> str:
    """Check stock from Supabase products table and return formatted string"""
    logger.info(f"Checking stock for: {model_name}")
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

            logger.info(
                f"Stock ALL: {len(in_stock_res.data)} in-stock, "
                f"{len(out_stock_res.data)} out-of-stock"
            )
            return json.dumps({
                "in_stock": in_stock_res.data,
                "out_of_stock": out_stock_res.data
            }, ensure_ascii=False)

        # 2. กรณีค้นหารายรุ่น — ใช้ ilike เพื่อ partial match (case-insensitive)
        search_query = model_name.strip()
        response = supabase_client.table("products").select("product_name, price, stock_quantity").ilike("product_name", f"%{search_query}%").execute()

        if response.data:
            logger.info(f"Stock found for '{model_name}': qty={response.data[0].get('stock_quantity')}")
            return json.dumps(response.data[0], ensure_ascii=False)

        # 3. หาไม่เจอจริงๆ
        logger.info(f"Stock not found for '{model_name}'")
        return json.dumps({"status": "not_found", "model": model_name}, ensure_ascii=False)

    except Exception:
        logger.exception(f"Supabase check_stock error for '{model_name}'")
        return json.dumps({"status": "error", "model": model_name}, ensure_ascii=False)

def web_search_tool(query: str) -> str:
    logger.info(f"Web search: '{query}'")
    try:
        response = tavily.search(query=query, max_results=3, search_depth="advanced")
        results = response.get("results", [])
        logger.info(f"Web search returned {len(results)} results")

        context = "ผลลัพธ์การค้นหาเว็บ:\n"
        for result in results:
            context += f"- {result['title']}: {result['content']}\n"
        return context
    except Exception:
        logger.exception("Web search failed")
        return "เกิดข้อผิดพลาดในการค้นหาเว็บ"
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
        logger.warning("Detected text-based tool call (model did not use native tool calling)")
        
        pattern = r'<tool_call>(.*?)ground'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            try:
                tool_data = json.loads(json_str)
                logger.info(f"Parsed text-based tool call: {tool_data.get('name', 'unknown')}")
                
                # Mock Data for OpenAI tool call structure
                class FakeToolCall:
                    def __init__(self, name, args):
                        self.id = "call_fake_123"
                        self.type = "function"
                        self.function = type('obj', (object,), {
                            'name': name,
                            'arguments': json.dumps(args)
                        })

                return [FakeToolCall(tool_data["name"], tool_data["arguments"])]
                
            except Exception:
                logger.exception("Failed to parse text-based tool call")
    return None