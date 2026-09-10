"""
Human-Judge Agreement & Inter-Rater Reliability Calibration.
Measures correlation, Cohen's Kappa, and agreement percentage between Human Annotations and the LLM Judge.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from src.data.dataset_loader import load_human_judge_sample
from src.eval.llm_judge import LLMJudge

def evaluate_human_judge_agreement() -> Dict[str, Any]:
    """
    Executes judge scoring on the 50 human-annotated calibration set samples
    and calculates statistical agreement metrics.
    """
    samples = load_human_judge_sample()
    judge = LLMJudge()

    human_overalls = []
    judge_overalls = []
    
    dim_names = ["groundedness", "actionability", "brand_tone", "safety_and_escalation", "conciseness"]
    dim_human = {d: [] for d in dim_names}
    dim_judge = {d: [] for d in dim_names}

    sample_comparisons = []

    for item in samples:
        h_scores = item["human_scores"]
        h_overall = item["human_overall_score"]
        
        # Run judge evaluation on reference reply as simulated candidate
        j_eval = judge.evaluate_reply(
            customer_text=item["customer_text"],
            generated_reply=item["reference_reply"],
            reference_reply=item["reference_reply"],
            intent=item["ground_truth_intent"],
            gt_intent=item["ground_truth_intent"],
            escalated=item["ground_truth_escalate"],
            gt_escalated=item["ground_truth_escalate"]
        )
        j_scores = j_eval["dimension_scores"]
        j_overall = j_eval["overall_score"]

        human_overalls.append(h_overall)
        judge_overalls.append(j_overall)

        for d in dim_names:
            dim_human[d].append(h_scores[d])
            dim_judge[d].append(j_scores[d])

        sample_comparisons.append({
            "id": item["id"],
            "customer_text": item["customer_text"],
            "human_overall": h_overall,
            "judge_overall": j_overall,
            "score_diff": round(abs(h_overall - j_overall), 2),
            "human_scores": h_scores,
            "judge_scores": j_scores
        })

    # Statistical correlation across overall composite scores
    pearson_r, p_val = pearsonr(human_overalls, judge_overalls)
    spearman_rho, s_pval = spearmanr(human_overalls, judge_overalls)

    # Calculate exact and adjacent agreement rates across all individual rating instances
    total_ratings = 0
    exact_matches = 0
    adjacent_matches = 0 # within +/- 1 score

    dim_metrics = {}
    for d in dim_names:
        h_arr = dim_human[d]
        j_arr = dim_judge[d]
        
        # Cohen's Kappa for the dimension
        kappa = cohen_kappa_score(h_arr, j_arr)
        # Weighted kappa
        w_kappa = cohen_kappa_score(h_arr, j_arr, weights="quadratic")

        exact = sum(1 for h, j in zip(h_arr, j_arr) if h == j)
        adj = sum(1 for h, j in zip(h_arr, j_arr) if abs(h - j) <= 1)
        n = len(h_arr)

        exact_matches += exact
        adjacent_matches += adj
        total_ratings += n

        dim_metrics[d] = {
            "cohen_kappa": round(float(kappa), 4) if not np.isnan(kappa) else 1.0,
            "quadratic_weighted_kappa": round(float(w_kappa), 4) if not np.isnan(w_kappa) else 1.0,
            "exact_agreement_pct": round((exact / n) * 100, 2),
            "adjacent_agreement_pct": round((adj / n) * 100, 2),
            "mean_human_score": round(float(np.mean(h_arr)), 2),
            "mean_judge_score": round(float(np.mean(j_arr)), 2)
        }

    overall_exact_pct = round((exact_matches / total_ratings) * 100, 2)
    overall_adj_pct = round((adjacent_matches / total_ratings) * 100, 2)

    return {
        "sample_size": len(samples),
        "total_evaluated_rating_pairs": total_ratings,
        "pearson_correlation_r": round(float(pearson_r), 4),
        "pearson_p_value": float(p_val),
        "spearman_rank_correlation_rho": round(float(spearman_rho), 4),
        "overall_exact_agreement_pct": overall_exact_pct,
        "overall_adjacent_agreement_pct": overall_adj_pct,
        "dimension_breakdown": dim_metrics,
        "sample_comparisons": sample_comparisons[:10] # Top 10 for display
    }
