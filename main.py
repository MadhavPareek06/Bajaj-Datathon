import os
import json
import base64
import logging
import io
import re
import time
import zipfile
import requests
import fitz  
import concurrent.futures
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
    # Preserving your specific model ID
    MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct" 
    MAX_WORKERS = 5
    TIMEOUT_SECONDS = 45
    USER_AGENT = "Mozilla/5.0 (BillExtractor/1.0)"

if not Config.GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY is missing. API calls will fail.")

# Global Client
client = Groq(api_key=Config.GROQ_API_KEY)

app = FastAPI(title="Bill Extraction API", version="2.0")

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

class PageLineItems(BaseModel):
    page_no: str
    page_type: str = "Bill Detail"
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None

class DocumentRequest(BaseModel):
    document: str

# --- 3. UTILITY SERVICES ---

class NetworkService:
    """Handles file downloading with robust fallback strategies."""
    
    @staticmethod
    def download_file(url: str) -> Tuple[bytes, str]:
        """
        Downloads file and detects mime type.
        Returns: (file_bytes, mime_type)
        """
        headers = {"User-Agent": Config.USER_AGENT}
        
        # Strategies for cleaning the URL
        url_strategies = [
            url, 
            url.replace(" ", "%20"), 
            unquote(url)
        ]

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
        
        raise HTTPException(status_code=400, detail="Failed to download document after multiple attempts.")

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
        """Robustly extracts JSON from LLM markdown response."""
        try:
            # Remove markdown code blocks
            text = re.sub(r"```json|```", "", text).strip()
            
            # Find the first { and last }
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            return {"bill_items": []}

# --- 4. CORE LOGIC SERVICES ---

class ExtractionService:
    """Handles interaction with Groq AI."""

    PROMPT_TEMPLATE = """
    You are a strict data extraction AI. Analyze this bill image.
    Extract line items into a JSON object with key 'bill_items'.
    Each item must have: 'item_name' (string), 'item_amount' (float), 'item_rate' (float), 'item_quantity' (float).
    Use 0.0 for missing numbers. Do not include markdown or explanations.
    """

    @staticmethod
    def extract_from_image(image_bytes: bytes, page_num: int) -> Tuple[Dict, Dict]:
        """Sends a single image to Groq."""
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
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
                    temperature=0.1,
                    max_tokens=2000,
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
                if "429" in str(e): # Rate limit
                    logger.warning(f"⚠️ Rate Limit Page {page_num}. Retrying...")
                    time.sleep(2)
                    continue
                logger.error(f"❌ Page {page_num} Error: {e}")
                break
        
        return {"bill_items": []}, {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    @staticmethod
    def _normalize_data(data: Dict) -> Dict:
        """Ensures the data structure matches our Pydantic models."""
        normalized = []
        raw_items = data.get("bill_items", [])
        
        if isinstance(raw_items, dict): 
            raw_items = [raw_items] # Handle edge case where LLM returns single dict

        for item in raw_items:
            normalized.append({
                "item_name": str(item.get("item_name", "Unknown")),
                "item_amount": float(item.get("item_amount", 0.0)),
                "item_rate": float(item.get("item_rate", 0.0)),
                "item_quantity": float(item.get("item_quantity", 0.0))
            })
        return {"bill_items": normalized}

class DocumentProcessor:
    """Orchestrates the processing of PDFs, ZIPs, and Images."""

    @staticmethod
    def process(file_content: bytes, mime_type: str) -> Tuple[List[PageLineItems], TokenUsage]:
        extracted_pages = []
        total_usage = TokenUsage()

        # Helper to aggregate results
        def collect_result(p_num: int, p_data: Dict, p_usage: Dict):
            if p_data.get("bill_items"):
                extracted_pages.append(PageLineItems(
                    page_no=str(p_num),
                    bill_items=[BillItem(**i) for i in p_data["bill_items"]]
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
                        # Recursive call? Simplified to just process PDF content here
                        tasks = DocumentProcessor._convert_pdf_to_images(pdf_data)
                        DocumentProcessor._execute_parallel(tasks, collect_result)
                    except Exception as e:
                        logger.error(f"Failed to process zip entry {pdf_file}: {e}")

        # 3. Handle Single Image
        else:
            data, usage = ExtractionService.extract_from_image(file_content, 1)
            collect_result(1, data, usage)

        # Sort by page number
        extracted_pages.sort(key=lambda x: int(x.page_no) if x.page_no.isdigit() else 999)
        return extracted_pages, total_usage

    @staticmethod
    def _convert_pdf_to_images(pdf_bytes: bytes) -> List[Tuple[bytes, int]]:
        """Converts PDF pages to image bytes."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            tasks = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=100)
                tasks.append((pix.tobytes("jpg"), i + 1))
            return tasks
        except Exception as e:
            logger.error(f"PDF Conversion Error: {e}")
            return []

    @staticmethod
    def _execute_parallel(tasks: List[Tuple[bytes, int]], callback):
        """Runs extraction in parallel threads."""
        if not tasks: return

        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Map task to (image_bytes, page_num)
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
        # Parse Body
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
            
        return APIResponse(
            is_success=True,
            token_usage=token_usage,
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_items
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