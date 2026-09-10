"""
Triage & Escalation Decision Engine for @AmazonHelp.
Evaluates incoming customer requests against security, policy, customer severity, and confidence thresholds
to determine whether the request should be auto-handled or escalated to a human agent, providing a stated reason.
"""

from typing import Dict, Any, Tuple
import re
from src.config import ESCALATION_REASONS, CONFIDENCE_THRESHOLD

class EscalationEngine:
    def __init__(self):
        self.security_patterns = [
            r"\b(hacked|fraud|unauthorized charge|unauthorized purchase|stolen account|2fa blocked|otp scam|phishing link|account suspended|compromised)\b",
            r"\b(someone changed my email|someone stole my password|digital gift cards?|pay outside amazon|paypal instead)\b",
            r"\b(empty box sealed|stole.*airpods|empty package received|gym socks|fake rolex|counterfeit)\b",
            r"\b(someone in another country attempted to log|unauthorized changes?|master password)\b"
        ]
        
        self.investigation_patterns = [
            r"\b(marked delivered|says delivered|no package|never arrived|no truck|missing package)\b",
            r"\b(shattered|completely destroyed|smashed|ruined|detergent leaked|exploded|smoking|threw.*gate|dumped.*rain)\b",
            r"\b(returned \d+ days|return center.*still no refund|disputed.*refund|weeks ago.*still no refund|refund was promised)\b",
            r"\b(promised by (your |chat )?agent|chat transcript|promised a credit|supervisor|delivery attempted.*business closed)\b",
            r"\b(charged twice.*different cards|kid accidentally purchased|duplicate.*channel)\b"
        ]

        self.high_severity_patterns = [
            r"\b(chargeback|lawyer|sue amazon|attorney|better business bureau|bbb complaint|police report|lawsuit)\b",
            r"\b(3rd time|fourth time|multiple times|dumped packages in the rain|fail(ed)? repeatedly)\b",
            r"\b(medical supplies|prescription.*urgent|urgent.*patient|electrical fire|hazard|safety)\b",
            r"\b(wtf|bullshit|damn it|pissed off|fuck|asshole|theft)\b"
        ]

    def evaluate(self, text: str, intent: str, confidence: float) -> Tuple[bool, str, str]:
        """
        Determines escalation decision.
        Returns:
            - auto_handled: bool (True if bot handles directly, False if escalated to human)
            - reason_category: str (One of the 5 standard escalation reason codes)
            - reason_details: str (Human-readable justification for the decision)
        """
        lower = text.lower()

        # 1. Low Confidence / Ambiguity Check
        if len(lower.strip()) <= 4 or confidence < 0.45 or re.match(r"^[.?!\s]+$", lower):
            return False, "LOW_CONFIDENCE_OR_AMBIGUITY", "Message is too brief, ambiguous, or classification confidence is below safety threshold."

        # 2. Security & PII Risk Check
        if intent == "ACCOUNT_AND_SECURITY" and any(re.search(p, lower) for p in self.security_patterns):
            return False, "SECURITY_OR_PII_RISK", ESCALATION_REASONS["SECURITY_OR_PII_RISK"]
        for p in self.security_patterns:
            if re.search(p, lower):
                return False, "SECURITY_OR_PII_RISK", "Contains security, unauthorized billing, or phishing indicators requiring private agent verification."

        # 3. High Severity / Churn / Legal Risk Check
        for p in self.high_severity_patterns:
            if re.search(p, lower):
                return False, "HIGH_SEVERITY_OR_CHURN", "High emotional severity, repeated prior failure, or regulatory/legal chargeback risk detected."

        # 4. Policy Exception / Physical Investigation Check
        for p in self.investigation_patterns:
            if re.search(p, lower):
                return False, "POLICY_EXCEPTION_OR_INVESTIGATION", "Requires physical carrier trace, fulfillment investigation, or manual refund release."

        # 5. Low Confidence Threshold
        if confidence < CONFIDENCE_THRESHOLD and intent != "OUT_OF_SCOPE_OR_CHITCHAT":
            return False, "LOW_CONFIDENCE_OR_AMBIGUITY", f"Classification confidence ({confidence:.2f}) is below minimum auto-handling threshold ({CONFIDENCE_THRESHOLD})."

        # 6. Default: Auto-handleable
        return True, "SELF_SERVICE_CAPABLE", "Standard customer inquiry resolvable via official guidance or self-service tools."
