import os
import json
import base64
import logging
import io
import re
import time
import zipfile
import requests
import fitz  # PyMuPDF
import concurrent.futures
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from urllib.parse import unquote
from typing import List, Optional, Tuple, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

# --- 1. CONFIGURATION & LOGGING ---
load_dotenv()

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct" 
   
    MAX_WORKERS = 5
    TIMEOUT_SECONDS = 45
    USER_AGENT = "Mozilla/5.0 (BillExtractor/2.1)"

if not Config.GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY is missing. API calls will fail.")

# Global Client
client = Groq(api_key=Config.GROQ_API_KEY)

app = FastAPI(title="Bill Extraction API", version="2.2")

# --- 2. DATA MODELS (SCHEMAS) ---
class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

class BillItem(BaseModel):
    item_name: str = Field(default="Unknown")
    item_amount: float = 0.0
    item_rate: float = 0.0
    item_quantity: float = 0.0

class DiscountItem(BaseModel):
    description: str = Field(default="Discount")
    amount: float = 0.0 
    type: str = Field(default="fixed", description="'percentage' or 'fixed'")

class PageLineItems(BaseModel):
    page_no: str
    page_type: str = "Bill Detail"
    bill_items: List[BillItem]
    discounts: List[DiscountItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int
    total_discount_count: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None

# --- 3. UTILITY SERVICES ---

class ImagePreprocessor:
    """Handles Image enhancement for Yellow/Carbon papers."""
    
    @staticmethod
    def optimize_for_ocr(image_bytes: bytes) -> bytes:
        """
        Fixes Yellow Backgrounds by extracting the Red Channel.
        """
        try:
            # 1. Open Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 2. Extract Red Channel
            # Yellow = Red(High) + Green(High) + Blue(Low).
            # By taking the Red channel, the Yellow background becomes White (255),
            # and the dark ink stays Dark.
            r, g, b = img.split()
            processed_img = r
            
            # 3. Increase Contrast (Stretch the histogram)
            processed_img = ImageOps.autocontrast(processed_img, cutoff=2)

            # 4. Sharpen (Helps with the cursive handwriting in your example)
            enhancer = ImageEnhance.Sharpness(processed_img)
            processed_img = enhancer.enhance(2.0) # 2.0 = double sharpness

            # 5. Save back to bytes
            output = io.BytesIO()
            processed_img.save(output, format="JPEG", quality=90)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}. Using original image.")
            return image_bytes

class NetworkService:
    """Handles file downloading."""
    
    @staticmethod
    def download_file(url: str) -> Tuple[bytes, str]:
        headers = {"User-Agent": Config.USER_AGENT}
        url_strategies = [url, url.replace(" ", "%20"), unquote(url)]

        for attempt_url in url_strategies:
            try:
                logger.info(f"⬇️ Downloading: {attempt_url}")
                response = requests.get(attempt_url, headers=headers, timeout=Config.TIMEOUT_SECONDS)
                response.raise_for_status()
                content = response.content
                mime_type = NetworkService._detect_mime_type(content, attempt_url)
                return content, mime_type
            except requests.RequestException:
                continue
        
        raise HTTPException(status_code=400, detail="Failed to download document.")

    @staticmethod
    def _detect_mime_type(content: bytes, url: str) -> str:
        if content.startswith(b"%PDF"): return "application/pdf"
        if content.startswith(b"PK"): return "application/zip"
        if url.lower().endswith(('.png', '.jpg', '.jpeg')): return "image/jpeg"
        return "application/octet-stream"

class ParserUtils:
    """Helper functions for text parsing."""
    
    @staticmethod
    def clean_json_markdown(text: str) -> Dict[str, Any]:
        try:
            text = re.sub(r"```json|```", "", text).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            return {"bill_items": [], "discounts": []}

# --- 4. CORE LOGIC SERVICES ---
class ExtractionService:
    """Handles interaction with Groq AI."""

    # Keep your existing prompt template...
    PROMPT_TEMPLATE = """
    You are a strict data extraction AI. Analyze this image.
    
    TASK:
    1. EXTRACT LINE ITEMS (Key 'bill_items'):
       - Look for rows with Quantity, Drug/Item Name, Batch, and Amount.
       - Fields: 'item_name', 'item_amount' (total), 'item_rate' (unit price), 'item_quantity'.
       - IMPORTANT: The 'Name of Drugs' might be cursive. Do your best to transcribe. 
       - Examples from this context might be 'Fluticone', 'Criz M', 'Bilastmon'.
    
    2. EXTRACT DISCOUNTS (Key 'discounts'):
       - Look for text like "10% Discount", "Less", "Saving".
       - Fields: 'description', 'amount' (absolute value), 'type' ('fixed' or 'percentage').
    
    3. CRITICAL RULES:
       - ONLY extract text that is visible. 
       - IF THE IMAGE IS BLANK OR UNREADABLE, RETURN EMPTY LISTS. DO NOT MAKE UP DATA.
       - Return valid JSON only.
    """

    @staticmethod
    def _safe_float(value: Any) -> float:
        """
        Robustly converts any value (None, string with units, weird chars) to float.
        """
        if value is None:
            return 0.0
        
        try:
            # 1. Try direct conversion
            return float(value)
        except (ValueError, TypeError):
            # 2. Handle strings like "1x24", "Rs. 500", "500.00/-"
            try:
                # Remove everything except digits and dots
                clean_str = re.sub(r'[^\d.]', '', str(value))
                
                # Handle cases with multiple dots like "1.2.3" -> take the first part
                if clean_str.count('.') > 1:
                    clean_str = clean_str.split('.')[0] + '.' + clean_str.split('.')[1]
                    
                if not clean_str:
                    return 0.0
                    
                return float(clean_str)
            except Exception:
                return 0.0

    @staticmethod
    def extract_from_image(image_bytes: bytes, page_num: int) -> Tuple[Dict, Dict]:
        """Sends a single image to Groq."""
        
        # 1. Preprocess Image (Red Channel Fix)
        optimized_image = ImagePreprocessor.optimize_for_ocr(image_bytes)
        
        base64_image = base64.b64encode(optimized_image).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_image}"

        for attempt in range(3):
            try:
                logger.info(f"🚀 Page {page_num}: Sending to Groq (Attempt {attempt+1})...")
                
                completion = client.chat.completions.create(
                    model=Config.MODEL_ID,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ExtractionService.PROMPT_TEMPLATE},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )

                usage = completion.usage
                usage_stats = {
                    "total_tokens": usage.total_tokens,
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens
                }

                raw_content = completion.choices[0].message.content
                data = ParserUtils.clean_json_markdown(raw_content)
                return ExtractionService._normalize_data(data), usage_stats

            except Exception as e:
                if "429" in str(e):
                    time.sleep(2)
                    continue
                logger.error(f"❌ Page {page_num} Error: {e}")
                break
        
        return {"bill_items": [], "discounts": []}, {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    @staticmethod
    def _normalize_data(data: Dict) -> Dict:
        """Ensures the data structure matches our Pydantic models."""
        
        # Normalize Line Items
        norm_items = []
        raw_items = data.get("bill_items", [])
        if isinstance(raw_items, dict): raw_items = [raw_items]
        
        for item in raw_items:
            norm_items.append({
                "item_name": str(item.get("item_name", "Unknown")),
                # Use _safe_float instead of float()
                "item_amount": ExtractionService._safe_float(item.get("item_amount")),
                "item_rate": ExtractionService._safe_float(item.get("item_rate")),
                "item_quantity": ExtractionService._safe_float(item.get("item_quantity"))
            })

        # Normalize Discounts
        norm_discounts = []
        raw_discounts = data.get("discounts", [])
        if isinstance(raw_discounts, dict): raw_discounts = [raw_discounts]

        for disc in raw_discounts:
            norm_discounts.append({
                "description": str(disc.get("description", "Discount")),
                "amount": abs(ExtractionService._safe_float(disc.get("amount"))),
                "type": str(disc.get("type", "fixed"))
            })

        return {"bill_items": norm_items, "discounts": norm_discounts}

class DocumentProcessor:
    """Orchestrates the processing of PDFs, ZIPs, and Images."""

    @staticmethod
    def process(file_content: bytes, mime_type: str) -> Tuple[List[PageLineItems], TokenUsage]:
        extracted_pages = []
        total_usage = TokenUsage()

        def collect_result(p_num: int, p_data: Dict, p_usage: Dict):
            # Only add page if it has items or discounts
            if p_data.get("bill_items") or p_data.get("discounts"):
                extracted_pages.append(PageLineItems(
                    page_no=str(p_num),
                    bill_items=[BillItem(**i) for i in p_data["bill_items"]],
                    discounts=[DiscountItem(**d) for d in p_data["discounts"]]
                ))
            total_usage.total_tokens += p_usage.get("total_tokens", 0)
            total_usage.input_tokens += p_usage.get("input_tokens", 0)
            total_usage.output_tokens += p_usage.get("output_tokens", 0)

        # 1. Handle PDF
        if "pdf" in mime_type:
            tasks = DocumentProcessor._convert_pdf_to_images(file_content)
            DocumentProcessor._execute_parallel(tasks, collect_result)

        # 2. Handle ZIP
        elif "zip" in mime_type:
            with zipfile.ZipFile(io.BytesIO(file_content)) as z:
                pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]
                for pdf_file in pdf_files:
                    try:
                        pdf_data = z.read(pdf_file)
                        tasks = DocumentProcessor._convert_pdf_to_images(pdf_data)
                        DocumentProcessor._execute_parallel(tasks, collect_result)
                    except Exception as e:
                        logger.error(f"Failed to process zip entry {pdf_file}: {e}")

        # 3. Handle Single Image
        else:
            data, usage = ExtractionService.extract_from_image(file_content, 1)
            collect_result(1, data, usage)

        extracted_pages.sort(key=lambda x: int(x.page_no) if x.page_no.isdigit() else 999)
        return extracted_pages, total_usage

    @staticmethod
    def _convert_pdf_to_images(pdf_bytes: bytes) -> List[Tuple[bytes, int]]:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            tasks = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                # DPI 200 is good balance for speed/accuracy
                pix = page.get_pixmap(dpi=200) 
                tasks.append((pix.tobytes("jpg"), i + 1))
            return tasks
        except Exception as e:
            logger.error(f"PDF Conversion Error: {e}")
            return []

    @staticmethod
    def _execute_parallel(tasks: List[Tuple[bytes, int]], callback):
        if not tasks: return
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(ExtractionService.extract_from_image, t[0], t[1]): t[1] 
                for t in tasks
            }
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    data, usage = future.result()
                    callback(page_num, data, usage)
                except Exception as e:
                    logger.error(f"Worker failed for page {page_num}: {e}")

# --- 5. API ENDPOINTS ---

@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: Request):
    try:
        try:
            body = await request.json()
            doc_url = body.get("document")
            if not doc_url: raise ValueError
        except:
            raise HTTPException(status_code=422, detail="Invalid JSON or missing 'document' field")

        # 1. Download
        file_bytes, mime_type = NetworkService.download_file(doc_url)
        
        # 2. Process
        pagewise_items, token_usage = DocumentProcessor.process(file_bytes, mime_type)
        
        # 3. Calculate Totals
        total_items = sum(len(p.bill_items) for p in pagewise_items)
        total_discounts = sum(len(p.discounts) for p in pagewise_items)
            
        return APIResponse(
            is_success=True,
            token_usage=token_usage,
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_items,
                total_discount_count=total_discounts
            )
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Unexpected API Error")
        return APIResponse(
            is_success=False, 
            token_usage=TokenUsage(), 
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)