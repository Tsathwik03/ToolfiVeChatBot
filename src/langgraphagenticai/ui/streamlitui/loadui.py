import streamlit as st
import os
from src.langgraphagenticai.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config=Config()
        self.user_controls={}

    def load_streamlit_ui(self):
        st.set_page_config(page_title="AI"+self.config.get_page_title(), layout="wide")
        st.title("AI "+self.config.get_page_title())

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()
            
            self.user_controls["selected_llm"] = st.selectbox("Select LLM Model", llm_options)

            if self.user_controls["selected_llm"] == "Groq":
                groq_model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Groq Model", groq_model_options)
                self.user_controls["GROQ_API_KEY"]=st.session_state["GROQ_API_KEY"]= st.text_input("API Key", type="password")
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("Please enter your Groq API Key to proceed.")

            self.user_controls["selected_usecase"] = st.selectbox("Select Use Case", usecase_options)

            if self.user_controls["selected_usecase"] in ["Chatbot With Web", "AI News"]:
                os.environ["TAVILY_API_KEY"]=self.user_controls["TAVILY_API_KEY"]=st.session_state["TAVILY_API_KEY"]= st.text_input("Tavily API Key", type="password")
                if not self.user_controls["TAVILY_API_KEY"]:
                    st.warning("Please enter your Tavily API Key to proceed.")
            
            if self.user_controls["selected_usecase"] == "AI News":
                st.subheader("News Explorer Options")
                # Removed button, only selectbox remains
                st.session_state.timeframe = st.selectbox(
                    "Select Time Frame",
                    ["Daily", "Weekly", "Monthly"],
                    index=0)
            
            if self.user_controls["selected_usecase"] == "RAG Chatbot":
                self.user_controls["uploaded_file"] = st.file_uploader(
                    "Upload a document (PDF or TXT)", type=["pdf", "txt"]
                )
                if not self.user_controls["uploaded_file"]:
                    st.warning("Please upload a document to chat with.")
            
        return self.user_controls