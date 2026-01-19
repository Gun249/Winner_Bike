from .logger import logger
from .setupGPU import device
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
    
logger.info("Setup Model Embedding")

model_embedding_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_embedding_name, trust_remote_code=True)
logger.info("Loaded tokenizer successfully")

model = AutoModel.from_pretrained(model_embedding_name, trust_remote_code=True,use_safetensors=True).to(device)
logger.info(f"Loaded model successfully | Running on: {next(model.parameters()).device}")
logger.info("Model Embedding is ready")


@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
    model_name=model_embedding_name
)

async def embedding_func(texts: list[str]) -> np.ndarray:
    logger.info(f"Embedding {len(texts)} texts using BGE-M3 model")
    if not texts:
        return np.zeros((0, 1024)) # ต้องแก้ dimension เป็น 1024 ให้ตรงกัน

    try:
        # BGE-M3 รองรับ Context ยาว 8192
        logger.info("Tokenizing texts...")
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=8192
        ).to(device)

        with torch.no_grad():
            logger.info("Generating embeddings...")
            outputs = model(**inputs)
            # BGE-M3 ใช้ค่าจาก Token แรก (CLS Token) เป็นตัวแทนประโยค
            # ไม่ต้องทำ Average Pool ให้ยุ่งยากเหมือน E5
            embeddings = outputs.last_hidden_state[:, 0]

            # Normalize เพื่อให้พร้อมสำหรับการค้นหา (Cosine Similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
    except Exception as e:
        logger.error(f"❌ Error in embedding_func: {e}")
        if "CUDA error" in str(e):
            logger.warning("⚠️ CUDA Error: Restart Session Required")
        raise