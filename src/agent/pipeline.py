"""
Unified Amazon Customer Support Agent Pipeline.
Coordinates classification, RAG retrieval, triage/escalation evaluation, and grounded reply generation.
"""

import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.agent.intent_classifier import IntentClassifier
from src.agent.retriever import HistoricalResolutionRetriever
from src.agent.escalation_engine import EscalationEngine
from src.agent.reply_generator import GroundedReplyGenerator

class AgentResponse(BaseModel):
    customer_text: str
    intent: str
    intent_confidence: float
    intent_probabilities: Dict[str, float]
    auto_handled: bool
    escalate_to_human: bool
    escalation_reason_category: str
    escalation_reason_details: str
    draft_reply: str
    retrieved_resolutions: List[Dict[str, Any]]
    processing_time_ms: float

class AmazonSupportAgent:
    def __init__(self):
        print("Initializing Amazon Support Agent components...")
        self.retriever = HistoricalResolutionRetriever(top_k=3)
        self.classifier = IntentClassifier()
        self.escalation_engine = EscalationEngine()
        self.reply_generator = GroundedReplyGenerator()
        print("Amazon Support Agent ready.")

    def process_tweet(self, customer_text: str) -> AgentResponse:
        """
        End-to-end processing of an incoming customer tweet.
        """
        start_time = time.perf_counter()

        # 1. Intent Classification
        intent, confidence, prob_dist = self.classifier.classify(customer_text)

        # 2. Historical Context Retrieval
        retrieved_contexts = self.retriever.retrieve(customer_text, top_k=3)

        # 3. Triage & Escalation Evaluation
        auto_handled, reason_category, reason_details = self.escalation_engine.evaluate(
            text=customer_text,
            intent=intent,
            confidence=confidence
        )
        escalate_to_human = not auto_handled

        # 4. Grounded Reply Generation
        draft_reply = self.reply_generator.generate(
            customer_text=customer_text,
            intent=intent,
            auto_handled=auto_handled,
            reason_category=reason_category,
            retrieved_contexts=retrieved_contexts
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return AgentResponse(
            customer_text=customer_text,
            intent=intent,
            intent_confidence=round(confidence, 4),
            intent_probabilities=prob_dist,
            auto_handled=auto_handled,
            escalate_to_human=escalate_to_human,
            escalation_reason_category=reason_category,
            escalation_reason_details=reason_details,
            draft_reply=draft_reply,
            retrieved_resolutions=retrieved_contexts,
            processing_time_ms=round(latency_ms, 2)
        )
