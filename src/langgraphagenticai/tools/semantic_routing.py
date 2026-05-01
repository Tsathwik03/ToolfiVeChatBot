# src/langgraphagenticai/tools/semantic_router.py

from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

class SemanticRouter:
    """
    Classifies query intent using cosine similarity against
    example phrases — no LLM call needed, instant routing.
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Define intent examples
        self.routes = {
            "ainews_rag": [
                "AI news from last month",
                "what happened in AI research",
                "saved articles about machine learning",
                "historical AI developments",
                "papers about transformers",
            ],
            "web_search": [
                "latest news today",
                "what happened this week",
                "current events",
                "real time information",
                "who won yesterday",
                "todays date",
                "live scores",
            ],
            "direct": [
                "what is",
                "explain",
                "how does it work",
                "define",
                "tell me about",
                "hello",
                "hi",
            ]
        }
        # Pre-compute embeddings for all examples
        self._route_embeddings = {
            route: self.embeddings.embed_documents(examples)
            for route, examples in self.routes.items()
        }

    def route(self, query: str) -> list[str]:
        """Return list of tool names needed for this query."""
        query_emb = np.array(self.embeddings.embed_query(query))

        scores = {}
        for route, embs in self._route_embeddings.items():
            # Average cosine similarity against all examples
            sims = [
                np.dot(query_emb, np.array(e)) /
                (np.linalg.norm(query_emb) * np.linalg.norm(np.array(e)))
                for e in embs
            ]
            scores[route] = max(sims)  # best match

        best_route = max(scores, key=scores.get)
        score = scores[best_route]

        # Low confidence → use LLM fallback
        if score < 0.4:
            return ["llm_fallback"]

        if best_route == "direct":
            return []  # no tools
        if best_route == "ainews_rag":
            return ["ainews_rag"]
        if best_route == "web_search":
            return ["tavily_search_results_json"]

        return []