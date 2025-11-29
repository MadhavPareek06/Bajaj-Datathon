import os
import json
import base64
import requests
import fitz  # PyMuPDF: The tool that converts PDF to Image
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Setup Hugging Face Client
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

client = InferenceClient(api_key=HF_TOKEN)

# --- CONFIGURATION ---
# Robust Model for bill reading
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct" 

app = FastAPI()

# --- Pydantic Models ---
class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

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

# --- Helpers ---
def download_file(url: str) -> bytes:
    clean_url = "".join(url.split())
    try:
        from urllib.parse import unquote
        clean_url = unquote(clean_url)
        response = requests.get(clean_url, timeout=15)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def analyze_single_image(image_bytes: bytes, page_num: int):
    """
    Sends a SINGLE image (page) to the AI for processing.
    """
    # Convert image to Base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{image_b64}"

    prompt = f"""
    Analyze this bill image (Page {page_num}). Extract line items strictly.
    
    CRITICAL RULES:
    1. Extract 'item_name', 'item_amount' (Net), 'item_rate', 'item_quantity'.
    2. IGNORE 'Sub-total', 'Total', 'Tax' lines to prevent double counting.
    3. Output PURE JSON only. No markdown.
    
    JSON Schema:
    {{
      "page_type": "Bill Detail",
      "bill_items": [
         {{"item_name": "Item A", "item_amount": 10.0, "item_rate": 5.0, "item_quantity": 2.0}}
      ]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )

        response_text = completion.choices[0].message.content
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text), completion.usage
    except Exception as e:
        print(f"Page {page_num} Error: {e}")
        return {"page_type": "Error", "bill_items": []}, None

def process_document(file_content: bytes, mime_type: str):
    extracted_pages = []
    total_tokens = {"total": 0, "input": 0, "output": 0}

    # CASE A: It is a PDF -> Convert to Images
    if "pdf" in mime_type.lower() or file_content.startswith(b"%PDF"):
        print("Detected PDF. Converting pages to images...")
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            # Loop through all pages
            for page_index in range(len(pdf_document)):
                page = pdf_document.load_page(page_index)
                pix = page.get_pixmap()
                image_bytes = pix.tobytes("png") # Convert page to PNG bytes
                
                # Send this page to AI
                print(f"Processing Page {page_index + 1}...")
                page_data, usage = analyze_single_image(image_bytes, page_index + 1)
                
                # Add page info
                extracted_pages.append({
                    "page_no": str(page_index + 1),
                    "page_type": page_data.get("page_type", "Unknown"),
                    "bill_items": page_data.get("bill_items", [])
                })
                
                # Aggregate tokens
                if usage:
                    total_tokens["total"] += usage.total_tokens
                    total_tokens["input"] += usage.prompt_tokens
                    total_tokens["output"] += usage.completion_tokens
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF Processing failed: {str(e)}")

    # CASE B: It is already an Image -> Process directly
    else:
        print("Detected Image. Processing directly...")
        page_data, usage = analyze_single_image(file_content, 1)
        extracted_pages.append({
            "page_no": "1",
            "page_type": page_data.get("page_type", "Unknown"),
            "bill_items": page_data.get("bill_items", [])
        })
        if usage:
            total_tokens["total"] += usage.total_tokens
            total_tokens["input"] += usage.prompt_tokens
            total_tokens["output"] += usage.completion_tokens

    return extracted_pages, total_tokens

# --- Endpoint ---
@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: DocumentRequest):
    file_bytes = download_file(request.document)
    
    # Determine type (basic check)
    if request.document.lower().endswith(".pdf"):
        mime_type = "application/pdf"
    else:
        mime_type = "image/png"
    
    try:
        pagewise_items, token_counts = process_document(file_bytes, mime_type)
        
        # Calculate reconciliation
        total_items = 0
        total_amt = 0.0
        
        for page in pagewise_items:
            items = page.get("bill_items", [])
            total_items += len(items)
            for item in items:
                total_amt += item.get("item_amount", 0.0)
                
        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=token_counts["total"],
                input_tokens=token_counts["input"],
                output_tokens=token_counts["output"]
            ),
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_items,
                reconciled_amount=round(total_amt, 2)
            )
        )
    except Exception as e:
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)