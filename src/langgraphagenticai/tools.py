"""
tools.py
--------
Defines the two core tools the agent can use:

  1. ainews_rag  – Retrieves from the local AINEWS FAISS vector store.
  2. web_search  – Searches the internet via Tavily for real-time info.

Both are defined with the @tool decorator so Groq's bind_tools() can
generate proper function-call schemas.
"""

import os
from functools import lru_cache

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from rag_indexer import build_or_load_index
from config import TAVILY_API_KEY


# ── Tool 1: AINEWS RAG ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_vectorstore():
    """Cached so the index is only loaded once per session."""
    return build_or_load_index()


@tool
def ainews_rag(query: str) -> str:
    """
    Search the local AI News knowledge base (AINEWS documents).
    Use this tool when the user asks about AI companies (OpenAI, Google DeepMind,
    Anthropic, Meta AI, etc.), AI models (GPT, Gemini, Claude, Llama, etc.),
    machine learning research, AI industry trends, or any topic likely covered
    in stored AI news articles.
    Input: a specific search query string.
    """
    try:
        vectorstore = _get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the AI News knowledge base for this query."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "AINEWS document")
            results.append(f"[Result {i} | Source: {source}]\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"AINEWS RAG error: {str(e)}"


# ── Tool 2: Tavily Web Search ─────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the internet for real-time, up-to-date information using Tavily.
    Use this tool when the user asks about current events, breaking news, live data,
    recent announcements, stock prices, sports scores, or any information that may
    have changed recently and is not in the local knowledge base.
    Input: a specific search query string.
    """
    try:
        if not TAVILY_API_KEY:
            return "Tavily API key not configured. Please add TAVILY_API_KEY to your .env file."

        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        searcher = TavilySearchResults(max_results=4)
        results = searcher.invoke(query)

        if not results:
            return "No results found from web search."

        formatted = []
        for i, r in enumerate(results, 1):
            url = r.get("url", "N/A")
            content = r.get("content", "").strip()
            formatted.append(f"[Result {i} | URL: {url}]\n{content}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {str(e)}"


# ── Tool Registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [ainews_rag, web_search]

TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS}

TOOL_DESCRIPTIONS = "\n".join(
    [f"  • {t.name}: {t.description.strip().splitlines()[0]}" for t in ALL_TOOLS]
)