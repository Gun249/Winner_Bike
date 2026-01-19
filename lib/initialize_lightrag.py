
from lightrag import LightRAG
from .logger import logger
from .embedding import embedding_func
from .llm_model import llm_model_func
import os

WORKING_DIR = "LightRAG_Data"



async def initialize_lightrag():
    logger.info("🚀 กำลัง Initialize LightRAG...")

    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            llm_model_name="gemini-2.5-flash",
            embedding_func=embedding_func,
            chunk_token_size =600,
            chunk_overlap_token_size=100,
        )
        logger.debug(f"Working directory: {WORKING_DIR}")
        logger.info("✅ Initialize LightRAG สำเร็จ")
        return rag
    except Exception as e:
        logger.error(f"❌ Error initializing LightRAG: {e}")
        raise