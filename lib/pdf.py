import asyncio
import PyPDF2
from .logger import logger
from pathlib import Path 
from lightrag import LightRAG
import re

async def read_pdf(pdf_path: str) -> str:
    logger.info(f"📄 กำลังอ่าน PDF: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            logger.info(f"จำนวนหน้า: {num_pages}")
            
            for i, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    page_text = Clean_Text(page_text)
                    text += page_text + "\n"

        
        logger.info(f"✅ อ่าน PDF เสร็จ: {len(text)} ตัวอักษรทั้งหมด")
        return text
    except Exception as e:
        logger.error(f"❌ Error reading PDF {pdf_path}: {e}")
        raise

async def load_pdfs_to_rag(rag: LightRAG, PDF_DIR: str):
    logger.info("📚 กำลังโหลด PDFs เข้า LightRAG...")
    print(f"Loading PDFs from directory: {PDF_DIR}")
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))

    print(pdf_files)

    if not pdf_files:
        logger.warning(f"⚠️ ไม่พบไฟล์ PDF ใน {PDF_DIR}")
        return
    
    logger.info(f"พบ {len(pdf_files)} ไฟล์ PDF")

    for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"📖 [{i}/{len(pdf_files)}] กำลังประมวลผล: {pdf_file.name}")
            try:
                text = await read_pdf(str(pdf_file))
                await rag.ainsert(text)
                logger.info(f"✅ เพิ่มเนื้อหาจาก {pdf_file.name} เข้า LightRAG สำเร็จ")
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file.name}: {e}")


def Clean_Text(text: str) -> str:

    if not text: return ""

    text = re.sub(r'^หน้า\s+[\d๑-๙]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'เล่ม.+ตอนที่.+', '', text)
    text = re.sub(r'มา\s?ตรา\s?', 'มาตรา ', text)


    thai_numbers = {
        '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4', '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
    }
    for thai_num, arabic_num in thai_numbers.items():
        text = text.replace(thai_num, arabic_num)
    
    text = text.replace('\u200b', '').replace('\xa0', '')  # ลบ Zero Width Space

    text = re.sub(r'\s+', ' ', text)  # แทนที่ช่องว่างหลายตัวด้วยช่องว่างเดียว

    text = text.strip()  # ลบช่องว่างที่ต้นและท้ายข้อความ
    return text