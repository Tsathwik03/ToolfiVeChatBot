"""
main.py
-------
Streamlit UI for the Agentic Chatbot.

Key design decisions:
  • NO manual agent/tool selection by the user.  The ToolFiVe filter node
    decides automatically which tools to invoke.
  • Every response shows an expandable ReAct trace so the user can see
    exactly which tool was called and what it returned.
  • The sidebar shows system status (API keys, index status) – not tool pickers.
"""

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from .pipeline.graph_builder import GraphBuilder
from .config import GROQ_API_KEY, TAVILY_API_KEY, AINEWS_DIR, FAISS_INDEX_PATH


# ── Page config ────────────────────────────────────────────────────────────────

def _page_config():
    st.set_page_config(
        page_title="Agentic AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.title("🤖 Agentic Chatbot")
        st.caption("ToolFiVe • LangGraph • Groq • FAISS • Tavily")

        st.divider()

        # ── System status ──────────────────────────────────────────────────
        st.subheader("⚙️ System Status")

        groq_ok   = bool(os.getenv("GROQ_API_KEY"))
        tavily_ok = bool(os.getenv("TAVILY_API_KEY"))

        from src.langgraphagenticai.tools.search_tool import get_vectorstore_status
        rag_status = get_vectorstore_status()

        ainews_ok  = rag_status["ainews_dir_exists"]
        file_count = rag_status["file_count"]
        index_ok   = rag_status["index_cached"] or rag_status["index_loaded"]

        st.markdown(
            f"{'🟢' if groq_ok   else '🔴'} **Groq LLM** "
            f"{'Connected' if groq_ok else 'Missing API key'}\n\n"
            f"{'🟢' if tavily_ok else '🔴'} **Tavily Search** "
            f"{'Connected' if tavily_ok else 'Missing API key'}\n\n"
            f"{'🟢' if ainews_ok else '🔴'} **AINEWS Folder** "
            f"{'Found' if ainews_ok else 'Not found'} "
            f"({file_count} file{'s' if file_count != 1 else ''} found)\n\n"
            f"{'🟢' if index_ok  else '🟡'} **FAISS Index** "
            f"{'Loaded ✓' if rag_status['index_loaded'] else ('Cached' if index_ok else 'Will build on first RAG query')}"
        )

        # Show which files are indexed
        if rag_status["files"]:
            with st.expander(f"📄 {file_count} indexed file(s)"):
                for f in rag_status["files"]:
                    st.caption(f"• {os.path.relpath(f)}")

        # Rebuild index button
        if ainews_ok and file_count > 0:
            if st.button("🔄 Rebuild RAG Index", use_container_width=True):
                with st.spinner("Rebuilding FAISS index from AINEWS/..."):
                    from src.langgraphagenticai.tools.search_tool import rebuild_vectorstore
                    vs = rebuild_vectorstore()
                    if vs:
                        st.success(f"✅ Index rebuilt from {file_count} files!")
                    else:
                        st.error("❌ Rebuild failed — check file formats")

        st.divider()

        # ── Architecture explanation ───────────────────────────────────────
        st.subheader("🧠 How it works")
        st.info(
            "**ToolFiVe Pipeline (auto)**\n\n"
            "1. 🔍 **Intent Recognition** – Semantic router picks tools\n\n"
            "2. ⚡ **ReAct Agent** – Reason → Act → Observe loop\n\n"
            "3. ✅ **Verifier** – Checks answer quality\n\n"
            "_You never need to pick a tool manually._"
        )

        st.subheader("🛠️ Available Tools")
        st.markdown(
            "- 🗃️ **ainews_rag** – searches local AINEWS documents (FAISS)\n"
            "- 🌐 **web_search** – searches internet via Tavily"
        )

        st.divider()

        # ── Conversation history ───────────────────────────────────────────
        st.subheader("📜 Conversation History")
        if st.session_state.get("chat_history"):
            for i, turn in enumerate(st.session_state.chat_history):
                with st.expander(f"Turn {i+1}: {turn['user'][:35]}…"):
                    st.markdown(f"**You:** {turn['user']}")
                    st.markdown(f"**Bot:** {turn['assistant'][:200]}…")
        else:
            st.caption("No conversation yet.")

        st.divider()

        # ── Clear chat ─────────────────────────────────────────────────────
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history  = []
            st.session_state.trace_history = []
            st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────

def _chat_area(graph):
    st.header("💬 Chat")
    st.caption(
        "Ask anything – the agent automatically decides whether to search the internet, "
        "query the AI news knowledge base, or answer from its own knowledge."
    )

    # ── Render existing conversation ───────────────────────────────────────
    for idx, turn in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(turn["user"])

        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

            trace = st.session_state.trace_history[idx] if idx < len(st.session_state.trace_history) else []
            if trace:
                _render_trace(trace, expanded=(idx == len(st.session_state.chat_history) - 1))

    # ── New user input ─────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask me anything…"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Reasoning with ToolFiVe pipeline…"):
                answer, trace = _run_graph(graph, user_input)

            st.markdown(answer)

            if trace:
                _render_trace(trace, expanded=True)

        st.session_state.chat_history.append({"user": user_input, "assistant": answer})
        st.session_state.trace_history.append(trace)


def _render_trace(trace: list, expanded: bool = False):
    """Render the step-by-step ReAct trace inside an expander."""
    with st.expander("🧠 View ReAct Trace (ToolFiVe steps)", expanded=expanded):
        for step in trace:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**{step['step']}**")
            with col2:
                st.code(step["content"], language="")
            st.divider()


def _run_graph(graph, user_input: str):
    try:
        history_messages = []
        for turn in st.session_state.chat_history:
            history_messages.append(HumanMessage(content=turn["user"]))
            history_messages.append(AIMessage(content=turn["assistant"]))

        history_messages.append(HumanMessage(content=user_input))

        result = graph.invoke(
            {
                "messages": history_messages,
                "filtered_tool_names": [],
                "verification_passed": False,
                "react_trace": [],
            }
        )

        final_msg = result["messages"][-1]
        answer = getattr(final_msg, "content", str(final_msg))
        trace = result.get("react_trace", [])
        return answer, trace

    except Exception as e:
        error_answer = (
            f"⚠️ An error occurred: {e}\n\n"
            "**Common fixes:**\n"
            "- Check that `GROQ_API_KEY` is set in your `.env` file.\n"
            "- Check that the `AINEWS/` folder exists and has `.pdf` or `.txt` files.\n"
            "- Run `pip install -r requirements.txt` to ensure all packages are installed."
        )
        return error_answer, []


# ── Entry point ────────────────────────────────────────────────────────────────

def load_langgraph_agenticai_app():
    _page_config()
    _sidebar()

    if "graph" not in st.session_state:
        with st.spinner("⚙️ Loading AI model and knowledge base…"):
            try:
                from .pipeline.graph_builder import GraphBuilder
                from .LLMS.groqllm import GroqLLM

                user_controls_input = {
                    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
                    "selected_groq_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                }

                llm = GroqLLM(user_controls_input).get_llm_model()
                builder = GraphBuilder(model=llm)
                st.session_state.graph = builder.setup_graph("Chatbot With Web")
            except Exception as e:
                st.error(f"Failed to initialise graph: {e}")
                st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "trace_history" not in st.session_state:
        st.session_state.trace_history = []

    _chat_area(st.session_state.graph)