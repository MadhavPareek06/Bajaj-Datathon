import os
import json
import base64
import requests
import fitz  # PyMuPDF
import concurrent.futures
import zipfile
import io
import re
import ast
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from urllib.parse import unquote

# Load environment variables
load_dotenv()

app = FastAPI()

# --- CONFIGURATION ---
# We use the 72B model for accuracy, but optimize image size for speed.
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct" 

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN is missing. API calls will fail.")

client = InferenceClient(api_key=HF_TOKEN)

# --- PYDANTIC MODELS ---
class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageLineItems(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int
    reconciled_amount: float

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None

class DocumentRequest(BaseModel):
    document: str

# --- HELPER FUNCTIONS ---

def download_file(url: str) -> bytes:
    """
    Smart Download: Tries RAW URL first (for Bots), then Cleaned URL (for Users).
    """
    print(f"⬇️ Downloading: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        # ATTEMPT 1: Try Raw URL (This fixes the Bot 400 Error)
        response = requests.get(url, headers=headers, timeout=45)
        response.raise_for_status()
        return response.content
    except Exception as e_raw:
        print(f"⚠️ Raw download failed ({e_raw}). Trying cleaned version...")
        
        # ATTEMPT 2: Try Cleaned URL (Fallback for manual testing)
        try:
            clean_url = unquote("".join(url.split()))
            response = requests.get(clean_url, headers=headers, timeout=45)
            response.raise_for_status()
            return response.content
        except Exception as e_clean:
            print(f"❌ All download attempts failed.")
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e_raw)}")

def robust_json_parse(text: str):
    """
    Robustly extracts JSON from AI text, handling markdown and common syntax errors.
    """
    try:
        # 1. Strip Markdown
        text = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Extract the main JSON object using Regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
            
        # 3. Try standard JSON parse
        return json.loads(text)
    except:
        # 4. Fallback: Python literal_eval (Handles single quotes vs double quotes)
        try:
            return ast.literal_eval(text)
        except:
            return None

def process_page_hf(image_bytes: bytes, page_num: int):
    """
    Sends a SINGLE image to Hugging Face. Optimized for Speed (JPEG + Low Res).
    """
    try:
        # SPEED OPTIMIZATION: Convert to JPEG (Smaller payload than PNG)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_image}"

        prompt = (
            "Analyze this bill image. Extract all line items into a JSON object.\n"
            "The JSON key must be 'bill_items'.\n"
            "Each item must have: 'item_name', 'item_amount', 'item_rate', 'item_quantity'.\n"
            "Ignore sub-totals or tax summaries to avoid double counting.\n"
            "Output ONLY the JSON."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]

        print(f"🚀 Page {page_num}: Sending to Hugging Face (72B Optimized)...")
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=2000,
            temperature=0.1
        )

        raw_content = completion.choices[0].message.content
        
        # Use robust parser
        data = robust_json_parse(raw_content)
        
        if data:
            return data
        else:
            print(f"❌ Page {page_num}: JSON Parse Failed.")
            return {"bill_items": []}

    except Exception as e:
        print(f"❌ Page {page_num} Error: {e}")
        return {"bill_items": []}

def process_document(file_content: bytes, mime_type: str):
    extracted_pages = []

    # CASE 1: ZIP File (Recursive)
    if "zip" in mime_type.lower() or file_content.startswith(b'PK'):
        print("📦 ZIP Detected. Extracting...")
        try:
            with zipfile.ZipFile(io.BytesIO(file_content)) as z:
                pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]
                for filename in pdf_files:
                    pdf_data = z.read(filename)
                    # Recursively process each PDF
                    pages = process_document(pdf_data, "application/pdf")
                    extracted_pages.extend(pages)
        except Exception as e:
            print(f"❌ ZIP Error: {e}")

    # CASE 2: PDF File
    elif "pdf" in mime_type.lower() or file_content.startswith(b"%PDF"):
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            tasks = []
            
            # Convert pages to images
            for i in range(len(doc)):
                page = doc.load_page(i)
                
                # SPEED OPTIMIZATION: DPI = 75
                # This makes the image 4x smaller in bytes -> Faster Upload -> Faster AI processing
                pix = page.get_pixmap(dpi=75) 
                
                # SPEED OPTIMIZATION: Use JPEG format
                img_bytes = pix.tobytes("jpg")
                tasks.append((img_bytes, i + 1))
            
            # Parallel Processing
            # 5 Workers is aggressive but safe for sequential PDF processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_page = {executor.submit(process_page_hf, t[0], t[1]): t[1] for t in tasks}
                
                for future in concurrent.futures.as_completed(future_to_page):
                    p_num = future_to_page[future]
                    try:
                        data = future.result()
                        extracted_pages.append({
                            "page_no": str(p_num),
                            "page_type": "Bill Detail",
                            "bill_items": data.get("bill_items", [])
                        })
                    except Exception:
                        pass
        except Exception as e:
            print(f"❌ PDF Error: {e}")

    # CASE 3: Image File
    else:
        # Assume it's an image, process directly
        data = process_page_hf(file_content, 1)
        extracted_pages.append({
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": data.get("bill_items", [])
        })

    # Sort results
    def safe_sort(x):
        try: return int(x["page_no"])
        except: return 999
    extracted_pages.sort(key=safe_sort)
    
    return extracted_pages

# --- MAIN API ENDPOINT ---
@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: DocumentRequest):
    try:
        # Download logic handles the URL messiness
        file_bytes = download_file(request.document)
        
        # Mime Type Logic
        if request.document.lower().endswith(".zip") or "zip" in request.document.lower():
             mime_type = "application/zip"
        elif request.document.lower().endswith(".pdf"):
             mime_type = "application/pdf"
        else:
             mime_type = "image/png"

        pagewise_items = process_document(file_bytes, mime_type)
        
        # Reconciliation Logic
        total_items = 0
        total_amt = 0.0
        for page in pagewise_items:
            items = page.get("bill_items", [])
            total_items += len(items)
            for item in items:
                # Handle possible string formatting in amount (e.g. "1,200.00")
                amt = item.get("item_amount", 0.0)
                if isinstance(amt, str):
                    try: amt = float(amt.replace(",", ""))
                    except: amt = 0.0
                total_amt += amt
                
        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(), # HuggingFace free API doesn't return usage stats easily
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_items,
                reconciled_amount=round(total_amt, 2)
            )
        )
    except Exception as e:
        return APIResponse(is_success=False, token_usage=TokenUsage(), error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)