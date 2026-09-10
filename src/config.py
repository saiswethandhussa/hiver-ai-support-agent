import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

BRAND_HANDLE = "@AmazonHelp"
BRAND_NAME = "Amazon Customer Service"

# Standardized Intent Taxonomy for @AmazonHelp
INTENT_TAXONOMY = {
    "DELIVERY_DELAY_OR_STATUS": {
        "description": "Customer inquiry about late shipments, delivery date estimates, tracking status, or carrier delays.",
        "typical_keywords": ["late", "delivery", "delivering", "tracking", "carrier", "where is my order", "arriving", "package delayed", "courier", "driver", "snowstorm", "in transit", "attempted delivery", "out for delivery", "mailroom"],
        "default_auto_handlable": True
    },
    "DAMAGED_OR_WRONG_ITEM": {
        "description": "Delivered package contains damaged items, broken parts, opened packaging, or completely incorrect merchandise.",
        "typical_keywords": ["broken", "damaged", "wrong item", "different product", "defective", "smashed", "shattered", "missing items"],
        "default_auto_handlable": False
    },
    "RETURN_AND_REFUND": {
        "description": "Questions regarding return drop-off process, return labels, refund status, or order cancellation before/after delivery.",
        "typical_keywords": ["refund", "return", "cancel order", "drop off", "ups dropoff", "kohls return", "money back", "return label"],
        "default_auto_handlable": True
    },
    "ACCOUNT_AND_SECURITY": {
        "description": "Issues accessing account, 2FA/OTP login failures, password reset, or unauthorized suspicious charges.",
        "typical_keywords": ["locked out", "login", "password", "otp", "2fa", "fraud", "unauthorized charge", "hacked", "account suspended"],
        "default_auto_handlable": False
    },
    "PRIME_AND_SUBSCRIPTION": {
        "description": "Prime membership fees, subscription renewal, accidental subscription charges, or digital media access (Prime Video/Music).",
        "typical_keywords": ["prime", "charged prime", "membership fee", "prime video", "subscription", "annual fee", "renewed without permission"],
        "default_auto_handlable": True
    },
    "PRODUCT_AND_ORDER_INQUIRY": {
        "description": "General product specs, warranty inquiries, invoice/receipt downloads, seller questions, or address changes before shipment.",
        "typical_keywords": ["invoice", "receipt", "warranty", "change address", "seller", "compatibility", "in stock", "order receipt"],
        "default_auto_handlable": True
    },
    "OUT_OF_SCOPE_OR_CHITCHAT": {
        "description": "General praise, unrelated commentary, social media banter, non-actionable complaints, or offensive spam.",
        "typical_keywords": ["great service", "thank you", "jeff bezos", "joke", "lol", "worst company ever", "random"],
        "default_auto_handlable": True
    }
}

INTENT_LIST = list(INTENT_TAXONOMY.keys())

# Escalation Reason Codes & Descriptions
ESCALATION_REASONS = {
    "SECURITY_OR_PII_RISK": "Issue involves sensitive authentication, OTPs, financial card data, or account compromise requiring verified private channels.",
    "POLICY_EXCEPTION_OR_INVESTIGATION": "Complex dispute (e.g. marked delivered but missing, high-value damage claim, refund pending > 14 days) requiring manual agent carrier trace.",
    "HIGH_SEVERITY_OR_CHURN": "Extreme customer frustration, threat of legal/chargeback action, or repeated unresolved customer contacts.",
    "LOW_CONFIDENCE_OR_AMBIGUITY": "Input message is too vague, ambiguous, multi-intent, or classification confidence is below the safety threshold (< 0.70).",
    "SELF_SERVICE_CAPABLE": "Standard question resolvable with known policy guidance, tracking help link, or self-service return workflow without human intervention."
}

# Model and API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

DEFAULT_EMBEDDING_DIM = 256
SIMILARITY_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.30
MAX_TWITTER_REPLY_CHARS = 280
