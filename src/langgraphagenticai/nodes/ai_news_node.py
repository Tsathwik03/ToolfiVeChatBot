from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
import os

class AINewsNode:
    def __init__(self, llm):
        self.tavily = TavilyClient()
        self.llm = llm
        self.state = {}

    def fetch_news(self, state: dict) -> dict:
        """
        Fetches news based on the specified topic and time frame.
        """
        content = state['messages'][-1].content
        lines = content.split('\n')
        topic = lines[0].replace("Topic: ", "").strip()
        frequency = lines[1].replace("Timeframe: ", "").strip().lower()
        
        self.state['topic'] = topic
        self.state['frequency'] = frequency
        
        time_range_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm', 'year': 'y'}
        t_range = time_range_map.get(frequency, 'd')

        try:
            # Broadened search parameters to guarantee results
            response = self.tavily.search(
                query=f"{topic} latest news",
                time_range=t_range,
                include_answer="advanced",
                max_results=10
            )
            self.state['news_data'] = response.get('results', [])
        except Exception as e:
            self.state['news_data'] = []
            print(f"Tavily Search Error: {e}")

        # Return empty dict so LangGraph doesn't duplicate the state messages
        return {}

    def summarize_news(self, state: dict) -> dict:
        """
        Summarizes the fetched news articles using the LLM.
        """
        news_items = self.state.get('news_data', [])
        topic = self.state.get('topic', 'Unknown')
        frequency = self.state.get('frequency', 'daily')
        
        # Failsafe if Tavily returns 0 articles
        if not news_items:
            self.state['summary'] = f"*No recent {frequency} news articles could be found for '{topic}'.*"
            return {}

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"Summarize the {frequency} news articles about '{topic}' into markdown format. For each article, include:\n"
             " - Date in ***YYYY-MM-DD*** format in IST Timezone\n"
             " - Concise sentence summary\n"
             " - Sort news by date wise (latest first)\n"
             " - Source URL as link\n"
             "Use format:\n"
             "### [Date]\n"
             "=[Summary](URL)"),
             ("user", "Articles\n{articles}")
        ])
        
        articles_str = "\n".join([
            f"Content: {item.get('content','')}\nURL: {item.get('url','')}\nDate: {item.get('published date','')}"
            for item in news_items
        ])
        
        try:
            response = self.llm.invoke(prompt_template.format_messages(articles=articles_str))
            self.state['summary'] = response.content
        except Exception as e:
            self.state['summary'] = f"*Error generating summary with LLM: {e}*"
            
        return {}

    def save_results(self, state: dict) -> dict:
        """
        Saves the summarized news to a markdown file.
        """
        topic = self.state.get('topic', 'Unknown')
        frequency = self.state.get('frequency', 'daily')
        safe_filename = topic.replace(" ", "_")[:20]
        summary = self.state.get('summary', '')
        
        os.makedirs('./AINEWS', exist_ok=True)
        filename = f"./AINEWS/{frequency}_{safe_filename}_summary.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {frequency.capitalize()} News Summary: {topic.title()}\n\n")
            f.write(summary)
        
        self.state['filename'] = filename
        return {}