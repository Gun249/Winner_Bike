import json
import os
import torch
import numpy as np
from .logger import logger
from lightrag.llm.gemini import gemini_model_complete 
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from dotenv import load_dotenv
load_dotenv()

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    is_keyword_task = False

    if keyword_extraction is True:
        is_keyword_task = True
    
    elif system_prompt and "Given the following text, extract" in system_prompt:
        is_keyword_task = True
    elif system_prompt and "Identify the high-level keywords" in system_prompt:
        is_keyword_task = True


    system_prompt_for_gemini = "You are an expert in analyzing text to extract key information. Your task is to identify and extract high-level keywords and main topics from the provided text. Focus on"

    if is_keyword_task:
        logger.info("ROUTER: Switching to GEMINI for Keyword Extraction") 
        try:
            return await gemini_model_complete(
                prompt,
                system_prompt=system_prompt_for_gemini, 
                history_messages=history_messages,
                api_key=os.getenv("GOOGLE_API_KEY"),
                model_name="gemini-2.5-flash", 
                keyword_extraction=keyword_extraction,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return "Error in keyword extraction"

    else:
        try:
            logger.info("ROUTER: Switching to GEMINI for Draft Response")

    
            original_rag_context = system_prompt if system_prompt else ""
        

            gemini_instruction = "Please read the following extensive context carefully and provide a concise and accurate draft response to the user's question based on that context."
        

            combined_system_prompt = f"{original_rag_context}\n\n{gemini_instruction}"

            logger.info(f"DEBUG Context Length: {len(combined_system_prompt)} chars")

            logger.info("CHAIN STEP 1: Gemini reading massive context...")
            draft_response = await gemini_model_complete(
                prompt,
                system_prompt=combined_system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("GOOGLE_API_KEY"),
                model_name="gemini-2.5-flash", 
                **kwargs
            )
            logger.info(" Gemini draft response received.")
            return draft_response
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return "ขออภัย ระบบขัดข้องชั่วคราว"
    

    