# src/langgraphagenticai/graph/graph_builder.py

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.basic_chatbot_nodes import BasicChatbotNode
from src.langgraphagenticai.nodes.ai_news_node import AINewsNode
from src.langgraphagenticai.nodes.rag_node import RAGNode
from src.langgraphagenticai.tools.search_tool import get_tools

import numpy as np


# ── Semantic Router ────────────────────────────────────────────────────────────

class SemanticRouter:
    """
    Classifies query intent using cosine similarity against example phrases.
    Fast, no LLM call needed. Returns tool names or 'llm_fallback'.
    """

    ROUTES = {
        "ainews_rag": [
            "AI news from last month",
            "what happened in AI research",
            "saved articles about machine learning",
            "historical AI developments",
            "papers about transformers",
            "previous AI announcements",
            "AI articles in knowledge base",
        ],
        "web_search": [
            "latest news today",
            "what happened this week",
            "current events",
            "real time information",
            "who won yesterday",
            "todays date",
            "live scores",
            "recent announcement",
            "breaking news",
            "stock price today",
        ],
        "direct": [
            "what is",
            "explain this concept",
            "how does it work",
            "define this term",
            "tell me about",
            "hello",
            "hi there",
            "thank you",
            "what can you do",
        ],
    }

    def __init__(self):
        self._embeddings = None
        self._route_embeddings = None

    def _get_embeddings(self):
        """Lazy-load embeddings model."""
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self._route_embeddings = {
                route: self._embeddings.embed_documents(examples)
                for route, examples in self.ROUTES.items()
            }
        return self._embeddings

    def route(self, query: str) -> tuple[list[str], str, str]:
        """
        Returns: (selected_tool_names, reason, method)
        method is 'semantic' or 'llm_fallback'
        """
        try:
            emb = self._get_embeddings()
            query_vec = np.array(emb.embed_query(query))

            scores = {}
            for route, vecs in self._route_embeddings.items():
                sims = [
                    np.dot(query_vec, np.array(v))
                    / (np.linalg.norm(query_vec) * np.linalg.norm(np.array(v)) + 1e-9)
                    for v in vecs
                ]
                scores[route] = max(sims)

            best_route = max(scores, key=scores.get)
            best_score = scores[best_route]

            # Low confidence → defer to LLM
            if best_score < 0.75:
                return ["llm_fallback"], "Low confidence — deferring to LLM", "llm_fallback"

            reason = f"Semantic match '{best_route}' (score={best_score:.2f})"

            if best_route == "direct":
                return [], reason, "semantic"
            elif best_route == "ainews_rag":
                return ["ainews_rag"], reason, "semantic"
            elif best_route == "web_search":
                return ["tavily_search_results_json"], reason, "semantic"

        except Exception as e:
            # If semantic router fails for any reason, fall back to LLM
            return ["llm_fallback"], f"Router error: {e}", "llm_fallback"

        return ["llm_fallback"], "Unknown route", "llm_fallback"


# ── Graph Builder ──────────────────────────────────────────────────────────────

class GraphBuilder:
    def __init__(self, model, user_controls=None):
        self.llm = model
        self.user_controls = user_controls or {}
        self.graph_builder = StateGraph(State)
        self._semantic_router = SemanticRouter()  # shared instance

    # ── Basic Chatbot ──────────────────────────────────────────────────────────
    def basic_chatbot_build_graph(self):
        node = BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot", node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)

    # ── ToolFiVe: Chatbot With Web + RAG ──────────────────────────────────────
    def chat_with_tools_build_graph(self):
        """
        Full ToolFiVe pipeline:
          Node 1 — tool_filter  : Semantic router + LLM fallback picks tools
          Node 2 — react_agent  : ReAct loop with ONLY filtered tools
          Node 3 — verifier     : LLM checks answer completeness vs CURRENT query
        """
        tools = get_tools()
        tool_map = {t.name: t for t in tools}
        valid_names = set(tool_map.keys())
        router = self._semantic_router

        # ── Node 1: Tool Filter ────────────────────────────────────────────
        def tool_filter_node(state: State) -> dict:
            # ✅ Always get the LATEST human message
            query = next(
                (m.content for m in reversed(state["messages"])
                 if isinstance(m, HumanMessage)),
                state["messages"][-1].content,
            )

            # ── Step 1: Fast semantic routing ──────────────────────────────
            selected, reason, method = router.route(query)

            # ── Step 2: LLM fallback if semantic router not confident ──────
            if "llm_fallback" in selected:
                tool_descriptions = "\n".join(
                    [f"- {t.name}: {t.description}" for t in tools]
                )
                prompt = f"""You are a tool selector for an AI assistant.

Available tools:
{tool_descriptions}

User query: {query}

Rules:
- Use 'ainews_rag' for questions about stored/historical AI news, research, or saved articles
- Use 'tavily_search_results_json' for real-time news, current events, live data, today's date
- Use BOTH if the query needs historical AND current information
- Use 'none' if the LLM can answer from its own knowledge (greetings, definitions, explanations)

Respond in EXACTLY this format (no extra text):
TOOLS: tool1,tool2
REASON: one sentence explanation"""

                response = self.llm.invoke(prompt)
                content = response.content.strip()

                selected, reason = [], "LLM fallback decision"
                for line in content.splitlines():
                    if line.startswith("TOOLS:"):
                        raw = line.replace("TOOLS:", "").strip()
                        if raw.lower() != "none":
                            selected = [t.strip() for t in raw.split(",")]
                    if line.startswith("REASON:"):
                        reason = line.replace("REASON:", "").strip()

                method = "LLM fallback"

            # ✅ Validate — only keep real tool names
            selected = [t for t in selected if t in valid_names]

            trace = list(state.get("react_trace", []))
            trace.append({
                "step": "🔍 Step 1 — Intent Recognition",
                "content": (
                    f"Query: {query}\n"
                    f"Routing method: {method}\n"
                    f"Selected tools: {selected if selected else ['none — direct LLM answer']}\n"
                    f"Reason: {reason}"
                ),
            })

            return {"filtered_tool_names": selected, "react_trace": trace}

        # ── Node 2: ReAct Agent ────────────────────────────────────────────
        def react_agent_node(state: State) -> dict:
            filtered_names = state.get("filtered_tool_names", [])
            filtered_tools = [t for t in tools if t.name in filtered_names]
            trace = list(state.get("react_trace", []))
            messages = list(state["messages"])

            # No tools needed — direct LLM answer
            if not filtered_tools:
                response = self.llm.invoke(messages)
                trace.append({
                    "step": "🤖 Step 2 — Direct Answer",
                    "content": (
                        "No tools needed — answered from LLM knowledge.\n\n"
                        + response.content
                    ),
                })
                return {"messages": messages + [response], "react_trace": trace}

            # Bind only the filtered tools to LLM
            llm_with_tools = self.llm.bind_tools(filtered_tools)

            # ReAct loop — max 5 iterations
            for iteration in range(5):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                # No tool calls → LLM has reached final answer
                if not getattr(response, "tool_calls", None):
                    trace.append({
                        "step": "🤖 Step 2 — Final Answer",
                        "content": response.content,
                    })
                    break

                # Execute each tool call
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_input = tc["args"]

                    trace.append({
                        "step": f"⚡ Step 2 — Action (iter {iteration + 1})",
                        "content": f"Tool: {tool_name}\nInput: {tool_input}",
                    })

                    if tool_name in tool_map:
                        try:
                            observation = tool_map[tool_name].invoke(tool_input)
                        except Exception as e:
                            observation = f"Tool error: {e}"
                    else:
                        observation = f"Unknown tool: {tool_name}"

                    trace.append({
                        "step": f"👁️ Step 2 — Observation (iter {iteration + 1})",
                        "content": str(observation)[:1500],
                    })

                    messages.append(
                        ToolMessage(
                            content=str(observation),
                            tool_call_id=tc["id"],
                        )
                    )

            return {"messages": messages, "react_trace": trace}

        # ── Node 3: Verifier ───────────────────────────────────────────────
        def verifier_node(state: State) -> dict:
            # ✅ Always get the LATEST human message (not first in history)
            query = next(
                (m.content for m in reversed(state["messages"])
                 if isinstance(m, HumanMessage)),
                "",
            )
            # ✅ Get the last AI message as the final answer
            final_answer = next(
                (m.content for m in reversed(state["messages"])
                 if isinstance(m, AIMessage)),
                state["messages"][-1].content,
            )
            trace = list(state.get("react_trace", []))

            prompt = f"""Did the assistant fully answer the user's question?

Question: {query}
Answer: {final_answer}

Respond in EXACTLY this format (no extra text):
VERDICT: PASSED
REASON: one sentence"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            verdict, reason = "PASSED", ""
            for line in content.splitlines():
                if line.startswith("VERDICT:"):
                    verdict = line.replace("VERDICT:", "").strip()
                if line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()

            trace.append({
                "step": f"✅ Step 3 — Verifier: {verdict}",
                "content": f"Verdict: {verdict}\nReason: {reason}",
            })

            return {
                "verification_passed": verdict == "PASSED",
                "react_trace": trace,
            }

        # ── Wire the graph ─────────────────────────────────────────────────
        self.graph_builder.add_node("tool_filter", tool_filter_node)
        self.graph_builder.add_node("react_agent", react_agent_node)
        self.graph_builder.add_node("verifier", verifier_node)

        self.graph_builder.add_edge(START, "tool_filter")
        self.graph_builder.add_edge("tool_filter", "react_agent")
        self.graph_builder.add_edge("react_agent", "verifier")
        self.graph_builder.add_edge("verifier", END)

    # ── AI News ────────────────────────────────────────────────────────────────
    def ai_news_builder_graph(self):
        ai_news_node = AINewsNode(self.llm)
        self.graph_builder.add_node("fetch_news", ai_news_node.fetch_news)
        self.graph_builder.add_node("summarize_news", ai_news_node.summarize_news)
        self.graph_builder.add_node("save_results", ai_news_node.save_results)
        self.graph_builder.set_entry_point("fetch_news")
        self.graph_builder.add_edge("fetch_news", "summarize_news")
        self.graph_builder.add_edge("summarize_news", "save_results")
        self.graph_builder.add_edge("save_results", END)

    # ── RAG Chatbot ────────────────────────────────────────────────────────────
    def rag_chatbot_build_graph(self):
        rag_node = RAGNode(self.llm)
        uploaded_file = self.user_controls.get("uploaded_file")
        if uploaded_file:
            rag_node.process_document(uploaded_file)
        self.graph_builder.add_node("retrieve", rag_node.retrieve)
        self.graph_builder.add_node("generate", rag_node.generate)
        self.graph_builder.add_edge(START, "retrieve")
        self.graph_builder.add_edge("retrieve", "generate")
        self.graph_builder.add_edge("generate", END)

    # ── Setup ──────────────────────────────────────────────────────────────────
    def setup_graph(self, usecase: str):
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        elif usecase == "Chatbot With Web":
            self.chat_with_tools_build_graph()
        elif usecase == "AI News":
            self.ai_news_builder_graph()
        elif usecase == "RAG Chatbot":
            self.rag_chatbot_build_graph()
        return self.graph_builder.compile()