# 🧾 AI-Powered Bill Extractor API

A robust document intelligence API that extracts line-item details from complex multi-page medical bills. It uses a Large Language Model (LLM) to parse tabular data, strictly enforcing mathematical reconciliation to avoid double-counting.

## 🚀 Key Features

* **Accuracy First:** Custom logic prevents the common "Double Counting" error by identifying and ignoring Section Headers (e.g., "Pharmacy Total") versus actual line items.
* **Math Reconciliation:** Automatically calculates a `reconciled_amount` by summing individual extracted items to validate against the bill's grand total.
* **Multimodal Analysis:** Uses Vision-Language Models (Qwen/Gemini) to "see" the bill layout, handling non-standard grids that break traditional OCR.
* **Standardized Output:** Returns strict JSON with extracted quantities, rates, and normalized names.

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python)
* **AI Model:** Qwen2.5-VL-72B (via Hugging Face) / Gemini 1.5 Flash
* **Validation:** Pydantic
* **Deployment:** Docker / Render / Railway

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd bill-extractor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```properties
    # Add your API Key (HuggingFace or Google)
    HF_TOKEN=hf_xxxx... 
    # OR
    GOOGLE_API_KEY=AIza...
    ```

4.  **Run the Server:**
    ```bash
    python main.py
    ```
    The API will start at `http://localhost:8000`.

## 🔌 API Usage

**Endpoint:** `POST /extract-bill-data`

**Request Body:**
```json
{
    "document": "[https://link-to-your-bill-image.png](https://link-to-your-bill-image.png)"
}