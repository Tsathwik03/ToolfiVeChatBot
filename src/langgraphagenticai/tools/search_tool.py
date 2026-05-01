# src/langgraphagenticai/tools/search_tool.py

import os
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
AINEWS_DIR       = os.getenv("AINEWS_DIR", "AINEWS/")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index/")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ All supported file extensions — NOW INCLUDES .md
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")

# ── Global vector store (lazy loaded once per process) ────────────────────────
_vectorstore = None


def _walk_ainews_dir(base_dir: str) -> list[str]:
    """
    Recursively walk base_dir and return all supported file paths.
    Supports: .pdf, .txt, .md
    """
    found = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                found.append(os.path.join(root, fname))
    return found


def _load_or_build_vectorstore():
    """
    Build FAISS index from all supported files inside AINEWS/ (recursively),
    or load a cached index if one already exists.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Load from cache ────────────────────────────────────────────────────
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            _vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"FAISS index loaded from cache: {FAISS_INDEX_PATH}")
            return _vectorstore
        except Exception as e:
            logger.warning(f"Failed to load cached index, rebuilding: {e}")

    # ── Build from documents ───────────────────────────────────────────────
    if not os.path.exists(AINEWS_DIR):
        logger.warning(f"AINEWS directory not found: {AINEWS_DIR}")
        return None

    # ✅ Recursively find all supported files including subfolders
    all_files = _walk_ainews_dir(AINEWS_DIR)
    logger.info(f"Found {len(all_files)} files in AINEWS/: {all_files}")

    if not all_files:
        logger.warning(
            f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) "
            f"found in AINEWS/ (including subfolders)"
        )
        return None

    docs = []
    for fpath in all_files:
        try:
            if fpath.lower().endswith(".pdf"):
                loaded = PyPDFLoader(fpath).load()
            else:
                # ✅ .txt AND .md both handled by TextLoader
                loaded = TextLoader(fpath, encoding="utf-8").load()
            docs.extend(loaded)
            logger.info(f"  Loaded {len(loaded)} pages from {fpath}")
        except Exception as e:
            logger.warning(f"  Could not load {fpath}: {e}")

    if not docs:
        logger.warning("All files failed to load — vectorstore not built")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    _vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index for next run
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    _vectorstore.save_local(FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

    return _vectorstore


def get_vectorstore_status() -> dict:
    """Returns a status dict for the Streamlit sidebar health check."""
    files = _walk_ainews_dir(AINEWS_DIR) if os.path.exists(AINEWS_DIR) else []
    return {
        "ainews_dir_exists": os.path.exists(AINEWS_DIR),
        "file_count": len(files),
        "files": files,
        "index_cached": os.path.exists(FAISS_INDEX_PATH),
        "index_loaded": _vectorstore is not None,
    }


def rebuild_vectorstore():
    """Force a full rebuild — call this if AINEWS/ contents change."""
    global _vectorstore
    _vectorstore = None
    import shutil
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    return _load_or_build_vectorstore()


# ── RAG Tool ───────────────────────────────────────────────────────────────────
@tool
def ainews_rag(query: str) -> str:
    """
    Search the local AINEWS knowledge base for AI news, research papers,
    and saved articles. Use this for questions about stored or historical
    AI news that may not be available on the internet.
    """
    vs = _load_or_build_vectorstore()
    if vs is None:
        return (
            "AINEWS knowledge base is not available. "
            f"Ensure the AINEWS/ folder contains supported files: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )

    results = vs.similarity_search(query, k=4)
    if not results:
        return "No relevant documents found in the AINEWS knowledge base for this query."

    output = []
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        output.append(f"[{i}] Source: {source}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)


# ── Tavily Web Search Tool ─────────────────────────────────────────────────────
def get_tavily_tool():
    """Live web search via Tavily — for real-time/current news queries."""
    return TavilySearchResults(
        max_results=5,
        description=(
            "Search the internet for real-time and current information. "
            "Use this for recent news, live events, current date/time, "
            "or anything not stored in the local knowledge base."
        ),
    )


# ── Combined tool list ─────────────────────────────────────────────────────────
def get_tools():
    """Return all tools available to the ToolFiVe agent."""
    tools = [ainews_rag]
    try:
        tools.append(get_tavily_tool())
    except Exception as e:
        logger.warning(f"Tavily tool not available: {e}")
    return tools


def create_tool_node(tools):
    return ToolNode(tools)