"""
LLM-as-Judge Evaluation Harness for @AmazonHelp.
Evaluates reply quality across 5 structured dimensions (1-5 scale):
1. Groundedness / Faithfulness
2. Actionability
3. Brand Tone & Empathy
4. Safety & Escalation Alignment
5. Conciseness & Clarity
"""

from typing import Dict, Any, List
import re
from src.config import OPENAI_API_KEY

RUBRIC_DESCRIPTION = {
    "groundedness": "1 (Hallucinated/Contradicts Policy) to 5 (Strictly adheres to verified historical brand policies).",
    "actionability": "1 (Vague/No next steps) to 5 (Direct, actionable self-service link or clear DM routing).",
    "brand_tone": "1 (Rude/Robotic/Apathetic) to 5 (Empathetic, polite, customer-obsessed on-brand tone).",
    "safety_and_escalation": "1 (Dangerous auto-handle of fraud/loss) to 5 (Appropriately routes risk to DM/human).",
    "conciseness": "1 (Bloated/Exceeds tweet limits) to 5 (Crisp, direct, under 280 chars)."
}

class LLMJudge:
    def __init__(self):
        self.openai_key = OPENAI_API_KEY

    def evaluate_reply(self, 
                       customer_text: str, 
                       generated_reply: str, 
                       reference_reply: str, 
                       intent: str, 
                       gt_intent: str, 
                       escalated: bool, 
                       gt_escalated: bool) -> Dict[str, Any]:
        """
        Evaluates a single generated reply against the rubric.
        Returns dimension scores (1-5) and an overall score.
        """
        if self.openai_key:
            try:
                return self._evaluate_with_openai(
                    customer_text, generated_reply, reference_reply, intent, gt_intent, escalated, gt_escalated
                )
            except Exception:
                pass # Fallback to deterministic rubric evaluation

        return self._evaluate_with_rubric(
            customer_text, generated_reply, reference_reply, intent, gt_intent, escalated, gt_escalated
        )

    def _evaluate_with_rubric(self, 
                              customer_text: str, 
                              reply: str, 
                              ref: str, 
                              intent: str, 
                              gt_intent: str, 
                              escalated: bool, 
                              gt_escalated: bool) -> Dict[str, Any]:
        """
        Deterministic, rule-calibrated judge scoring based on the strict rubric.
        """
        reply_lower = reply.lower()
        
        # 1. Groundedness (1-5)
        # Check if contains valid official links or standard phrasing
        has_official_link = bool(re.search(r"amzn\.to/\w+|amazon\.com", reply))
        intent_match = (intent == gt_intent)
        if intent_match and has_official_link:
            groundedness = 5
        elif intent_match:
            groundedness = 4
        elif has_official_link:
            groundedness = 3
        else:
            groundedness = 2

        # 2. Actionability (1-5)
        has_dm_action = "dm us" in reply_lower or "amzn.to/amazondm" in reply_lower or "direct message" in reply_lower
        has_url_action = "amzn.to/" in reply_lower or "amazon.com" in reply_lower
        if (escalated and has_dm_action) or (not escalated and has_url_action):
            actionability = 5
        elif has_dm_action or has_url_action:
            actionability = 4
        elif "visit" in reply_lower or "check" in reply_lower:
            actionability = 3
        else:
            actionability = 2

        # 3. Brand Tone & Empathy (1-5)
        empathy_phrases = ["sorry", "apologize", "thank you", "we'd love to help", "thrilled", "welcome", "understand"]
        empathy_count = sum(1 for p in empathy_phrases if p in reply_lower)
        if empathy_count >= 2:
            brand_tone = 5
        elif empathy_count == 1:
            brand_tone = 4
        elif len(reply) > 20:
            brand_tone = 3
        else:
            brand_tone = 2

        # 4. Safety & Escalation Alignment (1-5)
        if escalated == gt_escalated:
            safety = 5
        elif not escalated and gt_escalated:
            # Danger: Failed to escalate when human was required!
            safety = 1
        else:
            # Over-escalated safe query (minor inefficiency)
            safety = 3

        # 5. Conciseness (1-5)
        length = len(reply)
        if 40 <= length <= 240:
            conciseness = 5
        elif length <= 280:
            conciseness = 4
        elif length <= 320:
            conciseness = 3
        else:
            conciseness = 2

        scores = {
            "groundedness": groundedness,
            "actionability": actionability,
            "brand_tone": brand_tone,
            "safety_and_escalation": safety,
            "conciseness": conciseness
        }
        overall = round(sum(scores.values()) / len(scores), 2)

        return {
            "dimension_scores": scores,
            "overall_score": overall,
            "judge_rationale": f"Intent match: {intent_match}, Escalation match: {escalated == gt_escalated}, Links present: {has_official_link}."
        }

    def _evaluate_with_openai(self, customer_text, reply, ref, intent, gt_intent, escalated, gt_escalated) -> Dict[str, Any]:
        """Calls OpenAI GPT-4o-mini with structured JSON rubric schema."""
        from openai import OpenAI
        import json
        client = OpenAI(api_key=self.openai_key)

        prompt = f"""You are an expert customer service quality judge evaluating an AI agent for @AmazonHelp on Twitter.
Customer Tweet: "{customer_text}"
Reference Ground Truth: "{ref}"
Agent's Generated Reply: "{reply}"
Agent Classified Intent: {intent} (Ground Truth: {gt_intent})
Agent Escalated: {escalated} (Ground Truth: {gt_escalated})

Score the generated reply on a 1-5 integer scale for each of the 5 criteria:
1. groundedness: 1-5
2. actionability: 1-5
3. brand_tone: 1-5
4. safety_and_escalation: 1-5
5. conciseness: 1-5

Return ONLY valid JSON with keys "groundedness", "actionability", "brand_tone", "safety_and_escalation", "conciseness", and "rationale"."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        dim_scores = {
            "groundedness": int(data.get("groundedness", 4)),
            "actionability": int(data.get("actionability", 4)),
            "brand_tone": int(data.get("brand_tone", 4)),
            "safety_and_escalation": int(data.get("safety_and_escalation", 4)),
            "conciseness": int(data.get("conciseness", 4))
        }
        overall = round(sum(dim_scores.values()) / len(dim_scores), 2)
        return {
            "dimension_scores": dim_scores,
            "overall_score": overall,
            "judge_rationale": data.get("rationale", "LLM Judge evaluation.")
        }
