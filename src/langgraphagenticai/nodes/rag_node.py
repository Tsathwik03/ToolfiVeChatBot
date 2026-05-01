# src/langgraphagenticai/nodes/rag_node.py

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from src.langgraphagenticai.state.state import State

# ✅ Same model name as search_tool.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGNode:
    def __init__(self, llm):
        self.llm = llm
        self.vectorstore = None
        # ✅ Fixed: use full model name matching search_tool.py
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def process_document(self, uploaded_file):
        """Loads, splits, and embeds the uploaded document into FAISS."""
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{uploaded_file.name}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if tmp_file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path, encoding="utf-8")

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        finally:
            os.remove(tmp_file_path)

    def retrieve(self, state: State) -> dict:
        """Retrieves relevant context based on the latest user message."""
        # ✅ Get latest human message
        query = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            state["messages"][-1].content,
        )

        if not self.vectorstore:
            return {"context": "No document loaded or processed."}

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context}

    def generate(self, state: State) -> dict:
        """Generates answer using retrieved context + conversation history."""
        # ✅ Get latest human message
        query = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            state["messages"][-1].content,
        )
        context = state.get("context", "")

        # ✅ Build conversation history string (last 6 turns max)
        history_messages = [
            m for m in state["messages"]
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        # Exclude the current query (last human message)
        prior_turns = history_messages[:-1][-12:]  # last 6 turns = 12 messages

        history_str = ""
        if prior_turns:
            history_str = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in prior_turns
            ])

        prompt = (
            "You are a helpful assistant with access to a knowledge base.\n\n"
            + (f"Conversation history:\n{history_str}\n\n" if history_str else "")
            + f"Retrieved context from knowledge base:\n{context}\n\n"
            f"Current question: {query}\n\n"
            "Answer using the context above. If the context doesn't contain "
            "relevant information, say so clearly."
        )

        response = self.llm.invoke(prompt)
        return {"messages": [response]}