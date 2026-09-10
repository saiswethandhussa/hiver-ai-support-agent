"""
Intent Classification Engine for @AmazonHelp.
Combines semantic TF-IDF ngram modeling with calibrated probabilistic estimation and keyword priors.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.config import INTENT_TAXONOMY, INTENT_LIST, CONFIDENCE_THRESHOLD
from src.data.dataset_loader import load_knowledge_base

class IntentClassifier:
    def __init__(self):
        self.intents = INTENT_LIST
        self.model = self._train_model()

    def _train_model(self) -> Pipeline:
        kb_data = load_knowledge_base()
        X_train = [item["customer_query"] for item in kb_data]
        y_train = [item["intent"] for item in kb_data]

        # Add targeted keyword variations for rich boundary separation
        for intent, info in INTENT_TAXONOMY.items():
            for kw in info["typical_keywords"]:
                X_train.append(f"{kw}")
                y_train.append(intent)
                X_train.append(f"help with {kw}")
                y_train.append(intent)
                X_train.append(f"my {kw} problem with amazon")
                y_train.append(intent)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=12000, sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies incoming customer text into one of the 7 intents.
        Returns:
            - predicted_intent: str
            - confidence: float (0.0 to 1.0)
            - probability_distribution: Dict[str, float]
        """
        clean_text = text.strip()
        if not clean_text or len(clean_text) <= 2:
            return "OUT_OF_SCOPE_OR_CHITCHAT", 0.35, {i: 1.0/len(self.intents) for i in self.intents}

        # Predict raw probabilities
        probs = self.model.predict_proba([clean_text])[0]
        classes = self.model.classes_

        prob_dist = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
        
        # Explicit domain keyword prior
        lower_text = clean_text.lower()
        for intent, meta in INTENT_TAXONOMY.items():
            for kw in meta["typical_keywords"]:
                if kw in lower_text:
                    prob_dist[intent] = prob_dist.get(intent, 0.0) + 0.25

        # Normalize probabilities
        total_p = sum(prob_dist.values())
        norm_dist = {k: round(v / total_p, 4) for k, v in prob_dist.items()}

        sorted_intents = sorted(norm_dist.items(), key=lambda x: x[1], reverse=True)
        best_intent, top_prob = sorted_intents[0]
        second_prob = sorted_intents[1][1] if len(sorted_intents) > 1 else 0.0

        # Margin-calibrated confidence score
        confidence = round(top_prob - (0.5 * second_prob), 4)

        return best_intent, max(confidence, 0.10), norm_dist
