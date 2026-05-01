from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from src.langgraphagenticai.state.state import State

class ChatbotWithToolNode:
    """
    Basic Chatbot Tool-enabled Node
    """
    def __init__(self, model):
        self.llm = model

    def create_chatbot(self, tools):
        """
        Returns a LangGraph-compatible chatbot node
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State) -> dict:
            messages = state["messages"]

            response = llm_with_tools.invoke(messages)

            return {
                "messages": [response]
            }

        return chatbot_node
