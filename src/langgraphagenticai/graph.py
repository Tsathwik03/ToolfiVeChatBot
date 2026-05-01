"""
Run this script from your project root:
    python write_graph.py
It will overwrite src/langgraphagenticai/graph.py with the correct code.
"""

import os

content = r'''"""
graph.py - ToolFiVe Architecture (No ReAct Loop)
Node 1: tool_filter   - selects relevant tools
Node 2: tool_executor - runs tools, produces grounded summary
Node 3: verifier      - checks answer quality
"""

import json
import operator
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from .config import GROQ_API_KEY, GROQ_MODEL
from .tools import ALL_TOOLS, TOOL_DESCRIPTIONS, TOOL_REGISTRY


class AgentState(TypedDict):
    messages:            Annotated[list, operator.add]
    filtered_tool_names: List[str]
    verification_passed: bool
    react_trace:         Annotated[list, operator.add]


def _make_llm(temperature: float = 0) -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=temperature)


def extract_clean_content(raw) -> str:
    try:
        items = raw if isinstance(raw, list) else eval(str(raw))
        if not isinstance(items, list):
            raise ValueError("not a list")
        lines = []
        for item in items[:3]:
            if isinstance(item, dict):
                title   = item.get("title", "")
                content = item.get("content", "")
                content = re.sub(r"Image \d+", "", content)
                content = re.sub(r"\[.*?\]\(.*?\)", "", content)
                content = re.sub(r"#{1,6}\s*", "", content)
                content = re.sub(r"\n{3,}", "\n\n", content).strip()
                content = content[:700]
                lines.append(f"[Source: {title}]\n{content}")
        return "\n\n---\n\n".join(lines) if lines else str(raw)[:1000]
    except Exception:
        return re.sub(r"\n{3,}", "\n\n", str(raw)).strip()[:1500]


def tool_filter_node(state: AgentState) -> dict:
    llm            = _make_llm()
    user_query     = state["messages"][-1].content
    all_tool_names = [t.name for t in ALL_TOOLS]

    filter_prompt = f"""You are a tool-selection expert implementing the ToolFiVe methodology.

Available tools:
{TOOL_DESCRIPTIONS}

User query: "{user_query}"

Tool purposes:
- tavily_search_results_json : fetches LIVE information from the internet right now
- ainews_rag                 : searches LOCAL documents stored on disk about AI news

Priority rules (apply IN ORDER, stop at first match):
1. Query explicitly says "search the web", "look up online", "browse", "internet"
   -> tavily_search_results_json ONLY
2. Query asks about sports, weather, stocks, current events (non-AI topics)
   -> tavily_search_results_json ONLY
3. Query asks for "latest", "this week", "today", "current" AND is about AI topic
   -> BOTH tavily_search_results_json AND ainews_rag
4. Query asks about "your documents", "local files", "what do you have stored"
   -> ainews_rag ONLY
5. Query is about an AI topic but does NOT specifically need live/latest data
   -> ainews_rag ONLY
6. Pure math, greetings, or general knowledge LLM already knows
   -> empty list []

Examples:
- "Search the web for Gemini news"           -> ["tavily_search_results_json"]
- "who won yesterdays ipl match"             -> ["tavily_search_results_json"]
- "what is the weather today"                -> ["tavily_search_results_json"]
- "Latest OpenAI news"                       -> ["ainews_rag", "tavily_search_results_json"]
- "Latest OpenAI news from web and docs"     -> ["ainews_rag", "tavily_search_results_json"]
- "updates on GPT5"                          -> ["ainews_rag", "tavily_search_results_json"]
- "What is in your local AI documents?"      -> ["ainews_rag"]
- "What is 2+2?"                             -> []
- "Hello"                                    -> []

Respond ONLY with valid JSON (no markdown, no preamble):
{{"selected_tools": ["TOOL_NAME"], "reasoning": "one-sentence explanation"}}

Valid tool names: {all_tool_names}"""

    response = llm.invoke([HumanMessage(content=filter_prompt)])

    try:
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed    = json.loads(raw.strip())
        selected  = [t for t in parsed.get("selected_tools", []) if t in all_tool_names]
        reasoning = parsed.get("reasoning", "")
    except Exception:
        selected  = all_tool_names
        reasoning = "Fallback - using all tools (parse error)."

    trace_entry = {
        "step": "🔍 Step 1 — Tool Filter (ToolFiVe)",
        "content": (
            f"Query analysed: \"{user_query}\"\n"
            f"Tools selected: {selected if selected else ['(none — direct LLM answer)']}\n"
            f"Reasoning: {reasoning}"
        ),
    }

    return {
        "filtered_tool_names": selected,
        "react_trace": [trace_entry],
    }


def tool_executor_node(state: AgentState) -> dict:
    llm            = _make_llm()
    filtered_names = state["filtered_tool_names"]
    active_tools   = [t for t in ALL_TOOLS if t.name in filtered_names]
    user_query     = state["messages"][-1].content
    trace_entries  = []
    observations   = []

    # Case A: No tools — direct LLM answer
    if not active_tools:
        system   = SystemMessage(content=(
            "You are a knowledgeable AI assistant. "
            "Answer the user question clearly and concisely from your own knowledge."
        ))
        response = llm.invoke([system] + state["messages"])
        trace_entries.append({
            "step":    "💬 Step 2 — Direct LLM Response (no tools needed)",
            "content": response.content,
        })
        return {"messages": [response], "react_trace": trace_entries}

    # Case B: Execute each selected tool once
    for tool in active_tools:
        tool_name = tool.name
        tool_fn   = TOOL_REGISTRY.get(tool_name)

        trace_entries.append({
            "step":    f"⚡ Step 2 — Action: {tool_name}",
            "content": f"Tool input: {user_query}",
        })

        if tool_fn:
            try:
                result = tool_fn.invoke(user_query)
            except Exception as e:
                result = f"Tool execution error: {e}"
        else:
            result = f"Unknown tool: {tool_name}"

        clean = extract_clean_content(result)
        observations.append(f"=== Results from {tool_name} ===\n{clean}")

        trace_entries.append({
            "step":    f"👁️ Step 2 — Observation from {tool_name}",
            "content": clean[:1200],
        })

    # Case C: Grounded summarization — LLM sees ONLY the tool observations
    combined = "\n\n".join(observations)

    summary_prompt = f"""You are given raw search results retrieved for the query: "{user_query}"

RAW RESULTS:
{combined}

Instructions:
- List ONLY the specific headlines, dates, and facts that appear in the results above.
- Do NOT add any information from your own knowledge or training memory.
- Do NOT write a generic conclusion paragraph.
- Include dates exactly as they appear in the source.
- Use this exact format:

According to [Source Name]:
• [Date] — [Exact headline or fact from source]
• [Date] — [Exact headline or fact from source]

If the results contain no relevant information, respond with exactly:
"No relevant results were found for this query."
"""

    summary = llm.invoke([HumanMessage(content=summary_prompt)])

    trace_entries.append({
        "step":    "🤖 Step 2 — Final Answer (grounded summary)",
        "content": summary.content,
    })

    return {"messages": [summary], "react_trace": trace_entries}


def verifier_node(state: AgentState) -> dict:
    llm = _make_llm()

    user_query = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    last_msg     = state["messages"][-1]
    agent_answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    verify_prompt = f"""You are a strict quality-verification expert.

User asked: "{user_query}"
Agent answered: "{agent_answer}"

Mark FAILED if the answer:
- Contains vague filler: "rapidly evolving", "cutting-edge", "continuously updated and improved"
- Does NOT contain specific dates, headlines, or sourced facts
- Mixes up different companies or products
- Is a generic product description instead of specific recent news or facts

Mark PASSED if the answer:
- Contains specific dated headlines or sourced facts
- Directly addresses what the user asked
- Is grounded in retrieved content, not training memory
- For direct factual questions like math or greetings, a correct answer is PASSED

Respond ONLY with valid JSON (no markdown):
{{"passed": true_or_false, "reason": "one-sentence explanation"}}"""

    response = llm.invoke([HumanMessage(content=verify_prompt)])

    try:
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        passed = bool(parsed.get("passed", True))
        reason = parsed.get("reason", "")
    except Exception:
        passed = True
        reason = "Verification skipped (parse error)."

    status = "✅ PASSED" if passed else "⚠️  FAILED (answer may be incomplete)"

    return {
        "verification_passed": passed,
        "react_trace": [{
            "step":    "✅ Step 3 — Verifier (ToolFiVe)",
            "content": f"Status: {status}\nReason: {reason}",
        }],
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("tool_filter",   tool_filter_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("verifier",      verifier_node)
    graph.set_entry_point("tool_filter")
    graph.add_edge("tool_filter",   "tool_executor")
    graph.add_edge("tool_executor", "verifier")
    graph.add_edge("verifier",      END)
    return graph.compile()
'''

target = os.path.join("src", "langgraphagenticai", "graph.py")
os.makedirs(os.path.dirname(target), exist_ok=True)

with open(target, "w", encoding="utf-8") as f:
    f.write(content)

print(f"✅ Written to {target}")
print(f"   Size: {os.path.getsize(target)} bytes")
print("\nVerify with:")
print('  grep -n "def.*node" src/langgraphagenticai/graph.py')