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
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from urllib.parse import unquote, quote

# Load environment variables
load_dotenv()

app = FastAPI()

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
# We use the 7B-Instruct model via API because Render cannot run it locally
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct" 

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
    # reconciled_amount field removed as requested

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
    Hyper-Robust Download: Handles Raw, Encoded, and Spaces in URLs.
    """
    print(f"⬇️ Downloading: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # ATTEMPT 1: Try Raw URL (Works for pre-encoded Bot URLs)
    try:
        response = requests.get(url, headers=headers, timeout=45)
        response.raise_for_status()
        return response.content
    except Exception as e1:
        print(f"⚠️ Attempt 1 failed ({e1}). Trying to fix spaces...")
        
        # ATTEMPT 2: Fix Spaces (Common issue with 'Final Data')
        # We replace spaces with %20 but keep everything else
        try:
            # First unquote to get back to raw string (e.g. 'Final Data')
            # Then quote ONLY the path, not the query params? That's hard.
            # Easier strategy: Just replace literal spaces with %20
            fixed_url = url.replace(" ", "%20")
            print(f"🔄 Retrying with space-fixed URL: {fixed_url}")
            response = requests.get(fixed_url, headers=headers, timeout=45)
            response.raise_for_status()
            return response.content
        except Exception as e2:
            print(f"⚠️ Attempt 2 failed ({e2}). Trying full clean...")

            # ATTEMPT 3: Full Clean (Unquote everything)
            try:
                clean_url = unquote(url)
                # Re-encode only spaces? No, let requests handle it? 
                # Requests handles basic encoding, but let's try just the unquoted version
                # in case the input was double-encoded.
                print(f"🔄 Retrying with fully unquoted URL: {clean_url}")
                response = requests.get(clean_url, headers=headers, timeout=45)
                response.raise_for_status()
                return response.content
            except Exception as e3:
                print(f"❌ All download attempts failed.")
                raise HTTPException(status_code=400, detail=f"Download failed. Last error: {str(e3)}")

def robust_json_parse(text: str):
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)
    except:
        try:
            return ast.literal_eval(text)
        except:
            return None

def process_page_hf(image_bytes: bytes, page_num: int):
    # RETRY LOGIC for Rate Limits
    for attempt in range(3):
        try:
            # OPTIMIZATION: Convert to JPEG, Resize to reduce tokens
            # This mimics the efficiency of your new script but for API usage
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{base64_image}"

            # --- UPDATED PROMPT FROM YOUR NEW SCRIPT ---
            prompt = """You are an intelligent document processing assistant. Analyze this medical bill page and extract line items.
Strictly adhere to these rules:
1. Return ONLY the JSON object.
2. Extract text exactly as seen.
3. Return '0' for counts (we calculate them later).
4. Use the schema provided previously:
{
  "bill_items": [
    {
      "item_name": "string",
      "item_amount": float,
      "item_rate": float,
      "item_quantity": float
    }
  ]
}"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]

            print(f"🚀 Page {page_num}: Sending to Hugging Face (7B)...")
            
            # Using API (Not local model) to save RAM
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"} 
            )

            raw_content = completion.choices[0].message.content
            data = robust_json_parse(raw_content)
            
            if data:
                # Normalize keys if model hallucinates slightly different names
                normalized_items = []
                raw_items = data.get("bill_items", [])
                # Handle case where AI returns a dict instead of list
                if isinstance(raw_items, dict): raw_items = [raw_items]
                
                for item in raw_items:
                    normalized_items.append({
                        "item_name": str(item.get("item_name", "Unknown")),
                        "item_amount": float(item.get("item_amount", 0.0)),
                        "item_rate": float(item.get("item_rate", 0.0)),
                        "item_quantity": float(item.get("item_quantity", 0.0))
                    })
                return {"bill_items": normalized_items}
            else:
                print(f"❌ Page {page_num}: JSON Parse Failed.")
                return {"bill_items": []}

        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 10
                print(f"⚠️ Rate Limit on Page {page_num}. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ Page {page_num} Error: {e}")
                return {"bill_items": []}
                
    return {"bill_items": []}

def process_document(file_content: bytes, mime_type: str):
    extracted_pages = []

    # CASE 1: ZIP
    if "zip" in mime_type.lower() or file_content.startswith(b'PK'):
        print("📦 ZIP Detected.")
        try:
            with zipfile.ZipFile(io.BytesIO(file_content)) as z:
                pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]
                for filename in pdf_files:
                    try:
                        pdf_data = z.read(filename)
                        pages = process_document(pdf_data, "application/pdf")
                        extracted_pages.extend(pages)
                    except Exception as e:
                        print(f"Failed to process {filename}: {e}")
        except Exception as e:
            print(f"❌ ZIP Error: {e}")

    # CASE 2: PDF
    elif "pdf" in mime_type.lower() or file_content.startswith(b"%PDF"):
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            tasks = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                # Optimization: 75 DPI is enough for 7B model reading text
                pix = page.get_pixmap(dpi=75)
                img_bytes = pix.tobytes("jpg")
                tasks.append((img_bytes, i + 1))
            
            # Parallel Processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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

    # CASE 3: IMAGE
    else:
        data = process_page_hf(file_content, 1)
        extracted_pages.append({
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": data.get("bill_items", [])
        })

    def safe_sort(x):
        try: return int(x["page_no"])
        except: return 999
    extracted_pages.sort(key=safe_sort)
    return extracted_pages

# --- MAIN ENDPOINT ---
@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: Request):
    try:
        try:
            body = await request.json()
            doc_url = body.get("document")
            if not doc_url:
                raise HTTPException(status_code=422, detail="Missing 'document' field")
        except Exception:
             raise HTTPException(status_code=422, detail="Invalid JSON body")

        file_bytes = download_file(doc_url)
        
        # Determine mime-type from content bytes if URL is ambiguous
        if file_bytes.startswith(b"%PDF"):
             mime_type = "application/pdf"
        elif file_bytes.startswith(b"PK"):
             mime_type = "application/zip"
        else:
             mime_type = "image/png"

        pagewise_items = process_document(file_bytes, mime_type)
        
        total_items = 0
        total_amt = 0.0 # Kept for internal logic if needed, or can be removed if strictly not used
        for page in pagewise_items:
            items = page.get("bill_items", [])
            total_items += len(items)
            # Calculation logic kept in case you need it later, but not returned in response
            for item in items:
                amt = item.get("item_amount", 0.0)
                if isinstance(amt, str):
                    try: amt = float(amt.replace(",", ""))
                    except: amt = 0.0
                total_amt += amt
                
        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(), 
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_items,
                # reconciled_amount removed here
            )
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        return APIResponse(is_success=False, token_usage=TokenUsage(), error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)