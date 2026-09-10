"""
Automated Evaluation Metrics for Intent Classification, Escalation Triage, and Reply Generation.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import re

def compute_intent_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """Computes overall accuracy and macro/weighted precision, recall, and F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm,
        "classification_report": report
    }

def compute_escalation_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """Computes precision, recall, F1, specificity, and confusion matrix for escalation decisions."""
    y_true_int = [1 if x else 0 for x in y_true]
    y_pred_int = [1 if x else 0 for x in y_pred]

    acc = accuracy_score(y_true_int, y_pred_int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
    
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(float(specificity), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist()
    }

def compute_reply_text_metrics(generated_replies: List[str], reference_replies: List[str]) -> Dict[str, Any]:
    """
    Computes lexical overlap (ROUGE, token overlap), length compliance, and official link usage.
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    r1_scores, r2_scores, rl_scores = [], [], []
    valid_lengths = 0
    link_compliant = 0

    for gen, ref in zip(generated_replies, reference_replies):
        scores = scorer.score(ref, gen)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

        if len(gen) <= 280:
            valid_lengths += 1
        
        # Check if official Amazon shortlink is present
        if re.search(r"amzn\.to/\w+|amazon\.com", gen, re.IGNORECASE):
            link_compliant += 1

    n = len(generated_replies)
    return {
        "rouge1_f1": round(float(np.mean(r1_scores)), 4),
        "rouge2_f1": round(float(np.mean(r2_scores)), 4),
        "rougeL_f1": round(float(np.mean(rl_scores)), 4),
        "length_compliance_rate": round(valid_lengths / n, 4) if n > 0 else 0.0,
        "official_link_rate": round(link_compliant / n, 4) if n > 0 else 0.0
    }
