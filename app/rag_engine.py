import os
import lancedb
import requests
import pymupdf
import pytesseract
from PIL import Image
import io
import instaloader
import tempfile
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- ENVIRONMENT VARIABLES ---
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_FALLBACK = "qwen3.5:4b"
TESSERACT_PATH = os.getenv("TESSERACT_CMD", r'C:\Program Files\Tesseract-OCR\tesseract.exe')

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_dir = "./data/lancedb"
table_name = "rag_table"

# --- HELPERS ---
def get_best_model(requested_model: str = None) -> str:
    if requested_model and requested_model.lower() not in ["string", "none", ""]:
        return requested_model
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                return models[0]["name"]
    except Exception:
        pass
    return DEFAULT_FALLBACK

def ingest_text(text: str) -> str:
    if not text.strip():
        return "Error: No text found to ingest."
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    if not chunks:
        return "Error: No chunks were created."
    
    vectors = embeddings.embed_documents(chunks)
    data = [{"vector": v, "text": t} for v, t in zip(vectors, chunks)]
    
    db = lancedb.connect(db_dir)
    if table_name in db.table_names():
        tbl = db.open_table(table_name)
        tbl.add(data)
    else:
        db.create_table(table_name, data=data)
    return f"Successfully ingested {len(chunks)} chunks into the vector database."

# --- INGESTION METHODS ---
def ingest_pdf(file_path: str):
    text = ""
    try:
        with pymupdf.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if not page_text.strip():
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    return ingest_text(text)

def ingest_image(file_path: str):
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return ingest_text(text)
    except Exception as e:
        return f"Error reading image: {str(e)}"

def ingest_instagram(url: str):
    L = instaloader.Instaloader()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if "/p/" in url:
                shortcode = url.split("/p/")[1].split("/")[0]
            elif "/reel/" in url:
                shortcode = url.split("/reel/")[1].split("/")[0]
            else:
                return "Error: Invalid Instagram URL format."
            
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            text = ""
            
            if post.is_sidecar:
                for node in post.get_sidecar_nodes():
                    if not node.is_video:
                        img_data = requests.get(node.display_url).content
                        img = Image.open(io.BytesIO(img_data))
                        text += pytesseract.image_to_string(img) + "\n"
            else:
                if not post.is_video:
                    img_data = requests.get(post.url).content
                    img = Image.open(io.BytesIO(img_data))
                    text += pytesseract.image_to_string(img) + "\n"
            
            if not text.strip():
                return "Error: No text could be extracted from the Instagram post."
            return ingest_text(text)
        except Exception as e:
            return f"Error scraping Instagram: {str(e)}"

# --- MAIN QUERY FUNCTION ---
def query_rag(question: str, model_name: str = None, api_key: str = None, api_url: str = None):
    if not os.path.exists(db_dir):
        return "No documents ingested yet. Please upload a file first."

    db = lancedb.connect(db_dir)
    if table_name not in db.table_names():
        return "No documents ingested yet. Please upload a file first."
        
    tbl = db.open_table(table_name)
    query_vector = embeddings.embed_query(question)
    results = tbl.search(query_vector).limit(3).to_list()
    context = "\n\n".join([res["text"] for res in results])
    
    prompt = f"""Answer the question based ONLY on the following context.
Do not output your thinking process.
    
Context:
{context}
    
Question: {question}
Answer:"""

    # 1. Cloud API Logic
    if api_url and api_url.strip() != "":
        if not api_key:
            return "Error: API key is missing for the cloud model."
            
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Cloud API Error: {response.text}"
        except Exception as e:
            return f"Network Error: {str(e)}"

    # 2. Local Ollama Logic
    selected_model = get_best_model(model_name)
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": selected_model, 
                "prompt": prompt, 
                "stream": False,
                "think": False
            }
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Ollama Error: {response.text}"
    except Exception as e:
        return f"Ollama Connection Error: {str(e)}"

# --- PROMPT OPTIMIZER ---
def optimize_prompt(user_question: str):
    """Rewrites a messy prompt into a concise one and estimates tokens."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(user_question))
    except Exception:
        token_count = int(len(user_question) / 4)

    rewrite_prompt = f"""You are an expert prompt engineer. 
    Rewrite the following user question to be as concise, clear, and token-efficient as possible. 
    Do not answer the question. Only output the rewritten question.
    
    User Question: {user_question}
    Rewritten:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": get_best_model(), 
                "prompt": rewrite_prompt, 
                "stream": False,
                "think": False
            }
        )
        optimized_prompt = response.json().get("response", user_question).strip()
    except Exception:
        optimized_prompt = user_question 

    try:
        optimized_tokens = len(encoding.encode(optimized_prompt))
    except Exception:
        optimized_tokens = int(len(optimized_prompt) / 4)

    return {
        "original_prompt": user_question,
        "original_tokens": token_count,
        "optimized_prompt": optimized_prompt,
        "optimized_tokens": optimized_tokens,
        "savings_percentage": round(((token_count - optimized_tokens) / token_count) * 100, 2) if token_count > 0 else 0
    }