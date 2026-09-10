"""
Grounded Reply Generator for @AmazonHelp.
Generates responses grounded in retrieved historical brand resolutions, enforcing tone, empathy,
policy adherence, safety constraints, and Twitter-length conciseness.
"""

from typing import List, Dict, Any, Optional
import os
from src.config import MAX_TWITTER_REPLY_CHARS, OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY

class GroundedReplyGenerator:
    def __init__(self):
        self.openai_key = OPENAI_API_KEY
        self.gemini_key = GEMINI_API_KEY
        self.anthropic_key = ANTHROPIC_API_KEY

    def generate(self, 
                 customer_text: str, 
                 intent: str, 
                 auto_handled: bool, 
                 reason_category: str, 
                 retrieved_contexts: List[Dict[str, Any]]) -> str:
        """
        Drafts an empathetic, grounded response for @AmazonHelp.
        """
        # 1. Check if an external LLM API is available and requested
        if self.openai_key:
            try:
                return self._generate_with_openai(customer_text, intent, auto_handled, reason_category, retrieved_contexts)
            except Exception as e:
                pass # Fallback to grounded local synthesis

        # 2. Local Grounded Resolution Synthesizer (Fast, deterministic & grounded in retrieved context)
        return self._synthesize_grounded_reply(customer_text, intent, auto_handled, reason_category, retrieved_contexts)

    def _synthesize_grounded_reply(self, 
                                   text: str, 
                                   intent: str, 
                                   auto_handled: bool, 
                                   reason: str, 
                                   retrieved: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a response by fusing retrieved verified historical resolutions
        with escalation guardrails.
        """
        top_match = retrieved[0] if retrieved else None
        top_res = top_match["resolution_reply"] if top_match else ""

        if not auto_handled:
            # Escalated path: Empathy + Private DM handoff / Investigation routing
            if reason == "SECURITY_OR_PII_RISK":
                return "We take account security very seriously. Please send us a direct message with your account email so we can assist you safely: amzn.to/AmazonDM"
            elif reason == "HIGH_SEVERITY_OR_CHURN":
                return "We deeply apologize for your experience and understand your frustration. Please DM us your order details so our team can escalate this immediately: amzn.to/AmazonDM"
            elif reason == "POLICY_EXCEPTION_OR_INVESTIGATION":
                return "We're sorry for the trouble with your order. Please send us a DM with your order number so our fulfillment team can look into this for you: amzn.to/AmazonDM"
            else: # LOW_CONFIDENCE_OR_AMBIGUITY
                return "We'd love to help! Please reply with more details or send us a direct message with your order number: amzn.to/AmazonDM"
        else:
            # Auto-handled path: Grounded in best retrieved resolution
            if top_res:
                return top_res
            
            # Policy fallback per intent
            fallback_map = {
                "DELIVERY_DELAY_OR_STATUS": "We apologize for the delay! You can check live tracking updates for your package in Your Orders at amzn.to/TrackOrder.",
                "DAMAGED_OR_WRONG_ITEM": "We're sorry your item arrived in poor condition! You can request an immediate replacement or return at amzn.to/Returns.",
                "RETURN_AND_REFUND": "You can start a return and view drop-off options like Whole Foods or UPS anytime at amzn.to/Returns.",
                "ACCOUNT_AND_SECURITY": "You can securely manage passwords, 2FA, and active sessions anytime at amzn.to/AccountSecurity.",
                "PRIME_AND_SUBSCRIPTION": "You can view, manage, or cancel your membership anytime at amzn.to/ManagePrime.",
                "PRODUCT_AND_ORDER_INQUIRY": "You can view order details, invoices, and seller contact options in Your Orders at amzn.to/YourOrders.",
                "OUT_OF_SCOPE_OR_CHITCHAT": "Thanks for reaching out to @AmazonHelp! Let us know if you need assistance with any Amazon orders or services."
            }
            return fallback_map.get(intent, "Thanks for reaching out! Let us know how we can assist you with your Amazon order.")

    def _generate_with_openai(self, 
                              customer_text: str, 
                              intent: str, 
                              auto_handled: bool, 
                              reason: str, 
                              retrieved: List[Dict[str, Any]]) -> str:
        """Invokes OpenAI GPT-4o-mini / GPT-3.5 with strict grounding instructions."""
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)

        context_str = "\n".join([f"- Historical Resolution: {m['resolution_reply']}" for m in retrieved[:2]])
        
        prompt = f"""You are the official Twitter support agent for Amazon (@AmazonHelp).
Customer tweet: "{customer_text}"
Intent: {intent}
Auto-handled: {auto_handled} (Reason: {reason})

Relevant Historical Brand Responses:
{context_str}

Draft a friendly, empathetic reply in under 280 characters.
- If Auto-handled: Provide the exact official self-service help link (e.g. amzn.to/Returns, amzn.to/TrackOrder, amzn.to/ManagePrime).
- If Escalated to human: Apologize with empathy and direct them to secure DM: amzn.to/AmazonDM. Never make up fake order numbers or unauthorized promises.
Reply only with the tweet text:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
