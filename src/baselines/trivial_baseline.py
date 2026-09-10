"""
Baseline 1: Trivial Baseline.
A trivial customer support agent that:
- Always predicts the dataset majority intent (DELIVERY_DELAY_OR_STATUS).
- Uses a fixed, canned template response for all queries.
- Escalates using a naive single-keyword match ("human" or "agent").
"""

from typing import Dict, Any, List
import time

class TrivialBaselineAgent:
    def __init__(self):
        self.majority_intent = "DELIVERY_DELAY_OR_STATUS"
        self.canned_reply = "Thank you for contacting @AmazonHelp! Please send us a direct message with your order ID so we can look into this for you: amzn.to/AmazonDM"

    def process_tweet(self, customer_text: str) -> Dict[str, Any]:
        start = time.perf_counter()
        lower = customer_text.lower()
        
        # Naive keyword escalation
        escalate = "human" in lower or "agent" in lower or "representative" in lower
        reason = "HIGH_SEVERITY_OR_CHURN" if escalate else "SELF_SERVICE_CAPABLE"

        return {
            "customer_text": customer_text,
            "intent": self.majority_intent,
            "intent_confidence": 0.50,
            "auto_handled": not escalate,
            "escalate_to_human": escalate,
            "escalation_reason_category": reason,
            "escalation_reason_details": "Trivial keyword check triggered" if escalate else "Trivial default auto-handled",
            "draft_reply": self.canned_reply,
            "retrieved_resolutions": [],
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 2)
        }
