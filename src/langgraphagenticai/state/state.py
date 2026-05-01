# src/langgraphagenticai/state/state.py

from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    filtered_tool_names: list[str]    # set by tool_filter node
    verification_passed: bool          # set by verifier node
    react_trace: list[dict]            # trace steps for UI display