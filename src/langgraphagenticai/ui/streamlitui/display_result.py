import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message):
        self.usecase=usecase
        self.graph=graph
        self.user_message=user_message
        
    def display_result_on_ui(self):
        usecase=self.usecase
        graph=self.graph
        user_message=self.user_message
        
        if usecase=="Basic Chatbot":
            for event in graph.stream({'messages':("user",user_message)}):
                for value in event.values():
                    with st.chat_message("user"):
                        st.write(user_message)
                    with st.chat_message("assistant"):
                        st.write(value['messages'].content)
                        
        elif usecase=="Chatbot With Web":
            initial_state={"messages":[user_message]}
            res=graph.invoke(initial_state)
            for message in res['messages']:
                if type(message)==HumanMessage:
                    with st.chat_message("user"):
                        st.write(message.content)
                elif type(message)==AIMessage:
                    with st.chat_message("assistant"):
                        st.write(message.content)
                elif type(message)==ToolMessage:
                    with st.chat_message("assistant"):
                        st.write("Tool Call Start")
                        st.write(message.content)
                        st.write("Tool Call End")
                        
        elif usecase=="AI News":
            topic = self.user_message
            timeframe = st.session_state.timeframe
            # Bundle both values together to prevent index error
            query_msg = f"Topic: {topic}\nTimeframe: {timeframe}"
            
            with st.spinner(f"Fetching {timeframe.lower()} news for: {topic}..."):
                result=graph.invoke({"messages":[query_msg]})
                try:
                    safe_filename = topic.replace(" ", "_")[:20]
                    NEWS_PATH=f"./AINEWS/{timeframe.lower()}_{safe_filename}_summary.md"
                    with open(NEWS_PATH,"r", encoding="utf-8") as f:
                        markdown_content=f.read()
                    st.markdown(markdown_content,unsafe_allow_html=True)
                except FileNotFoundError:
                    st.error(f"Summary file not found: {NEWS_PATH}")
                except Exception as e:
                    st.error(f"An error occurred while displaying the summary: {str(e)}")

        elif usecase == "RAG Chatbot":
            initial_state = {"messages": [user_message]}
            res = graph.invoke(initial_state)
            
            with st.chat_message("user"):
                st.write(user_message)
            
            # The last message is the generated response from the LLM
            with st.chat_message("assistant"):
                st.write(res['messages'][-1].content)