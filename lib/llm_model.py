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
    
    # ---------------------------------------------------------
    # 1. DEBUG LOG: ดูว่า LightRAG ส่งอะไรมาบ้าง (สำคัญมาก)
    # ---------------------------------------------------------
    # logger.info(f"DEBUG CHECK -> keyword_extraction arg: {keyword_extraction}")
    # logger.info(f"DEBUG CHECK -> system_prompt starts with: {system_prompt[:50] if system_prompt else 'None'}")

    # ---------------------------------------------------------
    # 2. Logic การเช็คที่ถูกต้อง (Strict Mode)
    # ---------------------------------------------------------
    is_keyword_task = False

    # กรณีที่ 1: LightRAG ส่ง Flag มาบอกตรงๆ (เชื่อถือได้ที่สุด)
    if keyword_extraction is True:
        is_keyword_task = True
    
    # กรณีที่ 2: Fallback (เช็คเฉพาะกรณีที่ system_prompt ชัดเจนจริงๆ)
    # เราจะไม่เช็คแค่คำว่า "keywords" ลอยๆ เพราะอาจติดมาใน prompt ทั่วไปได้
    elif system_prompt and "Given the following text, extract" in system_prompt:
        is_keyword_task = True
    elif system_prompt and "Identify the high-level keywords" in system_prompt:
        is_keyword_task = True


    system_prompt_for_gemini = "You are an expert in analyzing text to extract key information. Your task is to identify and extract high-level keywords and main topics from the provided text. Focus on"

    # ---------------------------------------------------------
    # 3. Router
    # ---------------------------------------------------------
    if is_keyword_task:
        # >>>> ใช้ GEMINI (Logic/Extraction)
        logger.info("🤖 ROUTER: Switching to GEMINI for Keyword Extraction") 
        try:
            return await gemini_model_complete(
                prompt,
                system_prompt=system_prompt_for_gemini, # ตรวจสอบว่าตัวแปรนี้ถูก define ไว้หรือยัง ถ้าไม่มีให้ใช้ system_prompt ปกติ
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
        
        # คำสั่งเพิ่มเติมที่เราอยากบอก Gemini
            gemini_instruction = "Please read the following extensive context carefully and provide a concise and accurate draft response to the user's question based on that context."
        
        # รวมร่าง: Context เดิม + คำสั่งใหม่
            combined_system_prompt = f"{original_rag_context}\n\n{gemini_instruction}"

            logger.info(f"DEBUG Context Length: {len(combined_system_prompt)} chars")

            logger.info("🧠 CHAIN STEP 1: Gemini reading massive context...")
            # logger.info(f"Prompt for Gemini: {prompt}")
            draft_response = await gemini_model_complete(
                prompt,
                system_prompt=combined_system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("GOOGLE_API_KEY"),
                model_name="gemini-2.5-flash", 
                **kwargs
            )

            logger.info(f"Draft Response: {draft_response}")    

            logger.info("ROUTER: Switching to TYPHOON for Final Response")

            system_prompt_for_typhoon = """
                You are a “Technical Motorcycle & Parts Consultant”.

                Persona:
                You are knowledgeable, honest, and straightforward.
                You speak like an experienced motorcycle technician who genuinely wants to help customers.
                Your priority is helping customers, not selling.

                Mission:
                Provide clear, accurate, and practical answers that match exactly what the customer asks.
                Do not give extra explanations unless the customer explicitly asks for more details.

                Core Conversation Rule (Very Important):
                - Answer ONLY the customer’s current question.
                - Keep responses short, direct, and practical.
                - Do NOT explain specifications, features, or comparisons unless the customer asks.
                - Act like a real store staff replying in chat, not a reviewer or article writer.

                Follow-up Behavior:
                - If the customer asks a follow-up question, then explain clearly and honestly.
                - Focus on real-world usage instead of technical numbers.
                - Keep explanations concise and easy to understand.

                Strict Restrictions:
                - No hype, exaggeration, or emotional sales language.
                - No hard selling.
                - No references, citations, or the word “reference”.
                - No emojis.
                - Do NOT use overly formal Thai words such as “ท่าน”, “เรียนแจ้ง”, or “จึงเรียนมาเพื่อทราบ”.

                Language & Tone:
                - Always respond in Thai.
                - Refer to yourself as “ผม” or “ทางร้าน”.
                - Use natural, spoken Thai.
                - Keep it concise, clear, and professional — like a trusted mechanic or store staff.


            """
            
            refine_instruction = f"""
                Below is accurate raw information (Draft):
                "{draft_response}"

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

            return await openai_complete_if_cache(
                "typhoon-v2.5-30b-a3b-instruct",
                refine_instruction,
                system_prompt=system_prompt_for_typhoon,
                history_messages=history_messages,
                api_key=os.getenv("TYPHOON_API_KEY"),
                base_url="https://api.opentyphoon.ai/v1",
                max_tokens=4096,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Typhoon Error: {e}")
            return "ขออภัย ระบบขัดข้องชั่วคราว"
    

    