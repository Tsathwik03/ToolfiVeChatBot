import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Model Settings ────────────────────────────────────────────────────────────
# ✅ New - current and supported
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Fast, capable model on Groq
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── File Paths ────────────────────────────────────────────────────────────────
AINEWS_DIR       = "AINEWS"            # Folder with AI news PDFs/TXTs
FAISS_INDEX_PATH = "ainews_faiss_index"