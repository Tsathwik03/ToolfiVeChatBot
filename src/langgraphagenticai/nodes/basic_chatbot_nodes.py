from src.langgraphagenticai.state.state import State


class BasicChatbotNode:
    """
    A basic chatbot node that handles user input and generates responses.
    """
    def __init__(self,model):
        self.llm=model
    
    def process(self,state:State)->dict:
        """
        process the user input and generate a response using the LLM
        """
        return {"messages":self.llm.invoke(state['messages'])}