"""
Benchmark Runner for Comparative Evaluation of AI Customer Support Agents.
Executes the 200-sample Golden Evaluation Set across:
1. Trivial Baseline Agent
2. Simple Baseline Agent
3. Proposed Grounded AI Support Agent
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from src.config import ARTIFACTS_DIR, INTENT_LIST
from src.data.dataset_loader import load_golden_eval_set
from src.agent.pipeline import AmazonSupportAgent
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent
from src.eval.metrics import (
    compute_intent_metrics,
    compute_escalation_metrics,
    compute_reply_text_metrics
)
from src.eval.llm_judge import LLMJudge
from src.eval.human_agreement import evaluate_human_judge_agreement

def run_full_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING FULL BENCHMARK ON 200 GOLDEN EVALUATION SAMPLES")
    print("=" * 70)

    golden_set = load_golden_eval_set()
    judge = LLMJudge()

    models = {
        "Trivial Baseline (Majority + Canned)": TrivialBaselineAgent(),
        "Simple Baseline (Naive Bayes + Naive Rule)": SimpleBaselineAgent(),
        "Proposed Agent (Grounded RAG + Guardrailed Triage)": AmazonSupportAgent()
    }

    results = {}
    detailed_predictions = {}
    failure_cases = {m: [] for m in models}

    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}...")
        y_true_intent = []
        y_pred_intent = []
        y_true_esc = []
        y_pred_esc = []
        gen_replies = []
        ref_replies = []
        latencies = []
        judge_scores = []
        dim_scores_all = {
            "groundedness": [], "actionability": [], "brand_tone": [],
            "safety_and_escalation": [], "conciseness": []
        }

        preds_log = []

        for sample in golden_set:
            q = sample["customer_text"]
            gt_intent = sample["ground_truth_intent"]
            gt_esc = sample["ground_truth_escalate"]
            ref_reply = sample["reference_reply"]
            tier = sample["difficulty_tier"]

            # Run model prediction
            t0 = time.perf_counter()
            resp = model.process_tweet(q)
            lat = (time.perf_counter() - t0) * 1000

            # Unpack response
            if isinstance(resp, dict):
                p_intent = resp["intent"]
                p_esc = resp["escalate_to_human"]
                p_reason = resp.get("escalation_reason_category", "UNKNOWN")
                p_reply = resp["draft_reply"]
            else:
                p_intent = resp.intent
                p_esc = resp.escalate_to_human
                p_reason = resp.escalation_reason_category
                p_reply = resp.draft_reply

            y_true_intent.append(gt_intent)
            y_pred_intent.append(p_intent)
            y_true_esc.append(gt_esc)
            y_pred_esc.append(p_esc)
            gen_replies.append(p_reply)
            ref_replies.append(ref_reply)
            latencies.append(lat)

            # Run Judge
            j_eval = judge.evaluate_reply(
                customer_text=q,
                generated_reply=p_reply,
                reference_reply=ref_reply,
                intent=p_intent,
                gt_intent=gt_intent,
                escalated=p_esc,
                gt_escalated=gt_esc
            )
            judge_scores.append(j_eval["overall_score"])
            for d, s in j_eval["dimension_scores"].items():
                dim_scores_all[d].append(s)

            # Check if failure
            is_intent_fail = (p_intent != gt_intent)
            is_esc_fail = (p_esc != gt_esc)
            if is_intent_fail or is_esc_fail or j_eval["overall_score"] < 3.0:
                failure_cases[model_name].append({
                    "id": sample["id"],
                    "customer_text": q,
                    "difficulty_tier": tier,
                    "gt_intent": gt_intent,
                    "pred_intent": p_intent,
                    "gt_escalate": gt_esc,
                    "pred_escalate": p_esc,
                    "escalation_reason": p_reason,
                    "generated_reply": p_reply,
                    "reference_reply": ref_reply,
                    "judge_overall": j_eval["overall_score"],
                    "failure_type": "Intent Mismatch" if is_intent_fail else ("Escalation Error" if is_esc_fail else "Low Quality Score")
                })

            preds_log.append({
                "id": sample["id"],
                "customer_text": q,
                "gt_intent": gt_intent,
                "pred_intent": p_intent,
                "gt_escalate": gt_esc,
                "pred_escalate": p_esc,
                "generated_reply": p_reply,
                "judge_score": j_eval["overall_score"],
                "latency_ms": round(lat, 2)
            })

        # Compute metric aggregates
        intent_res = compute_intent_metrics(y_true_intent, y_pred_intent, INTENT_LIST)
        esc_res = compute_escalation_metrics(y_true_esc, y_pred_esc)
        reply_res = compute_reply_text_metrics(gen_replies, ref_replies)

        mean_judge = round(float(np.mean(judge_scores)), 2)
        dim_means = {d: round(float(np.mean(vals)), 2) for d, vals in dim_scores_all.items()}

        results[model_name] = {
            "intent_metrics": {
                "accuracy": intent_res["accuracy"],
                "macro_f1": intent_res["macro_f1"],
                "weighted_f1": intent_res["weighted_f1"],
                "confusion_matrix": intent_res["confusion_matrix"]
            },
            "escalation_metrics": {
                "accuracy": esc_res["accuracy"],
                "precision": esc_res["precision"],
                "recall": esc_res["recall"],
                "f1_score": esc_res["f1_score"],
                "specificity": esc_res["specificity"],
                "false_positives": esc_res["false_positives"],
                "false_negatives": esc_res["false_negatives"]
            },
            "reply_quality": {
                "rouge1_f1": reply_res["rouge1_f1"],
                "rouge2_f1": reply_res["rouge2_f1"],
                "rougeL_f1": reply_res["rougeL_f1"],
                "official_link_rate": reply_res["official_link_rate"],
                "length_compliance_rate": reply_res["length_compliance_rate"]
            },
            "judge_quality": {
                "composite_score_out_of_5": mean_judge,
                "dimensions": dim_means
            },
            "latency": {
                "mean_ms": round(float(np.mean(latencies)), 2),
                "p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "p99_ms": round(float(np.percentile(latencies, 99)), 2)
            },
            "failure_count": len(failure_cases[model_name])
        }

        detailed_predictions[model_name] = preds_log

    # 4. Human-Judge Agreement Analysis
    print("\nRunning Human-Judge Agreement Calibration on 50 samples...")
    agreement_results = evaluate_human_judge_agreement()

    # Consolidate complete report artifact
    benchmark_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(golden_set),
        "intents_evaluated": INTENT_LIST,
        "model_results": results,
        "human_judge_agreement": agreement_results,
        "top_failure_modes_proposed_agent": failure_cases["Proposed Agent (Grounded RAG + Guardrailed Triage)"][:10],
        "detailed_predictions": detailed_predictions
    }

    artifact_path = ARTIFACTS_DIR / "benchmark_results.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2, ensure_ascii=False)
    print(f"\n-> Saved full benchmark results to {artifact_path.name}")

    return benchmark_payload

if __name__ == "__main__":
    run_full_benchmark()
