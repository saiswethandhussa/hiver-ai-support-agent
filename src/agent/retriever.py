"""
Historical Resolution Retriever (RAG Engine) for @AmazonHelp.
Indexes historical resolution QA pairs using hybrid TF-IDF ngram semantic representations
and retrieves the top-K most similar historical resolutions to ground draft replies.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data.dataset_loader import load_knowledge_base

class HistoricalResolutionRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.kb_items = load_knowledge_base()
        self.queries = [item["customer_query"] for item in self.kb_items]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.queries)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar historical resolutions.
        Returns a list of dicts with match details and similarity scores.
        """
        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices sorted descending
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            item = self.kb_items[idx]
            results.append({
                "id": item["id"],
                "intent": item["intent"],
                "historical_query": item["customer_query"],
                "resolution_reply": item["resolution_reply"],
                "similarity_score": round(score, 4)
            })
        return results

    def get_grounding_context(self, query: str, top_k: int = 2) -> str:
        """Formats retrieved historical resolutions into a clean grounding prompt block."""
        matches = self.retrieve(query, top_k=top_k)
        lines = []
        for i, m in enumerate(matches, 1):
            lines.append(f"Example {i} [Intent: {m['intent']}] (Similarity: {m['similarity_score']:.2f}):")
            lines.append(f"  Customer: {m['historical_query']}")
            lines.append(f"  Brand Reply: {m['resolution_reply']}")
        return "\n".join(lines)
