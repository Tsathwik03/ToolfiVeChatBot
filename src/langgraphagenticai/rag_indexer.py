"""
rag_indexer.py
--------------
Builds (or loads from disk) a FAISS vector index from all PDF and TXT files
inside the AINEWS/ directory.  Called once at startup; subsequent runs load
the cached index from disk.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import AINEWS_DIR, FAISS_INDEX_PATH, EMBEDDING_MODEL


def _load_documents():
    """Load all PDFs and TXT files from the AINEWS directory."""
    docs = []

    # ── Load PDFs ─────────────────────────────────────────────────────────
    if any(f.endswith(".pdf") for f in os.listdir(AINEWS_DIR) if os.path.isfile(os.path.join(AINEWS_DIR, f))):
        pdf_loader = DirectoryLoader(
            AINEWS_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
        )
        docs.extend(pdf_loader.load())

    # ── Load TXT files ────────────────────────────────────────────────────
    if any(f.endswith(".txt") for f in os.listdir(AINEWS_DIR) if os.path.isfile(os.path.join(AINEWS_DIR, f))):
        txt_loader = DirectoryLoader(
            AINEWS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
        )
        docs.extend(txt_loader.load())

    return docs


def build_or_load_index() -> FAISS:
    """
    Returns a FAISS retriever backed by the AINEWS document corpus.
    Builds the index on first run; loads from disk on subsequent runs.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Load cached index ────────────────────────────────────────────────
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"[RAG] Loading cached FAISS index from '{FAISS_INDEX_PATH}'")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # ── Build index from scratch ─────────────────────────────────────────
    print(f"[RAG] Building FAISS index from '{AINEWS_DIR}' ...")

    if not os.path.exists(AINEWS_DIR):
        raise FileNotFoundError(
            f"AINEWS directory not found at '{AINEWS_DIR}'. "
            "Please create it and add AI news PDF/TXT files."
        )

    docs = _load_documents()

    if not docs:
        raise ValueError(
            f"No documents found in '{AINEWS_DIR}'. "
            "Add .pdf or .txt files and restart."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[RAG] Index built with {len(chunks)} chunks and saved.")

    return vectorstore