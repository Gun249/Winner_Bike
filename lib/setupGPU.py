import torch
from .logger import logger

logger.info("Setup GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Detected Device: {device}")