"""
Baseline 2: Simple Baseline Agent.
A simple machine learning agent that:
- Uses a basic CountVectorizer + Multinomial Naive Bayes classifier for intent classification.
- Uses basic sentiment / naive polarity heuristic for escalation decision.
- Uses fixed, ungrounded static templates per intent without historical resolution RAG retrieval.
"""

from typing import Dict, Any, List
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.data.dataset_loader import load_knowledge_base

class SimpleBaselineAgent:
    def __init__(self):
        self.model = self._train_simple_model()
        self.static_templates = {
            "DELIVERY_DELAY_OR_STATUS": "Please check your tracking information in Your Orders at amazon.com.",
            "DAMAGED_OR_WRONG_ITEM": "You can return damaged items by going to amazon.com/returns.",
            "RETURN_AND_REFUND": "For refunds, please allow 3-5 business days after return receipt.",
            "ACCOUNT_AND_SECURITY": "Please reset your password under Your Account settings.",
            "PRIME_AND_SUBSCRIPTION": "You can view Prime membership details in your account.",
            "PRODUCT_AND_ORDER_INQUIRY": "Please refer to the product details page on our website.",
            "OUT_OF_SCOPE_OR_CHITCHAT": "Thank you for contacting Amazon support!"
        }

    def _train_simple_model(self) -> Pipeline:
        kb = load_knowledge_base()
        X = [item["customer_query"] for item in kb]
        y = [item["intent"] for item in kb]
        pipeline = Pipeline([
            ("counts", CountVectorizer(stop_words="english")),
            ("nb", MultinomialNB())
        ])
        pipeline.fit(X, y)
        return pipeline

    def process_tweet(self, customer_text: str) -> Dict[str, Any]:
        start = time.perf_counter()
        
        # 1. Simple Naive Bayes intent classification
        predicted_intent = self.model.predict([customer_text])[0]
        
        # 2. Simple naive sentiment / anger escalation check
        lower = customer_text.lower()
        negative_words = ["angry", "terrible", "worst", "hate", "lawyer", "stolen", "hacked", "police", "scam", "broken", "ridiculous"]
        escalate = any(w in lower for w in negative_words)
        reason = "HIGH_SEVERITY_OR_CHURN" if escalate else "SELF_SERVICE_CAPABLE"

        # 3. Static template reply (ungrounded)
        if escalate:
            draft_reply = "We apologize for the trouble. Please DM us your order number so we can look into this: amzn.to/AmazonDM"
        else:
            draft_reply = self.static_templates.get(predicted_intent, "Please visit amazon.com for help.")

        return {
            "customer_text": customer_text,
            "intent": predicted_intent,
            "intent_confidence": 0.60,
            "auto_handled": not escalate,
            "escalate_to_human": escalate,
            "escalation_reason_category": reason,
            "escalation_reason_details": "Simple negative keyword detector" if escalate else "Simple rule pass",
            "draft_reply": draft_reply,
            "retrieved_resolutions": [],
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 2)
        }
