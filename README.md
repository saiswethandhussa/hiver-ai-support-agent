# ⚡ @AmazonHelp AI Support Agent

> **Hiver SDE Intern — Take-Home Assignment**  
> An AI Customer Support Agent for `@AmazonHelp` built on real Twitter customer service data with **grounded RAG replies**, **safety triage / escalation reasoning**, and **rigorous proof of trustworthiness**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproduction Time](https://img.shields.io/badge/Reproduction-%3C%2030%20seconds-brightgreen.svg)]()

---

## 🚀 Quickstart: Reproduce Headline Results in < 15 Minutes

The entire benchmark runs out-of-the-box in **< 30 seconds** without requiring external cloud API keys or heavy GPU setup.

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/hiver-support-agent.git
cd hiver-support-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Headline Benchmark (< 30 Seconds)
```bash
python reproduce_results.py
```
This executes the tri-model evaluation on the **200-sample hand-curated Golden Evaluation Set**, evaluates the **5-dimension LLM Judge**, computes **Human-Judge agreement calibration statistics**, and prints formatted tables.

### 3. Launch the Interactive Web Dashboard
```bash
python run_demo.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to explore:
- 💬 **Live Agent Console:** Test any incoming tweet with real-time classification, RAG retrieval context, and triage reasoning.
- 📊 **Benchmark & Baselines View:** Quantitative comparison across all metrics.
- 🎯 **Golden Eval Set Explorer:** Filter the 200 curated examples by intent, escalation status, or difficulty tier.
- ⚖️ **Human-Judge Calibration:** View inter-rater agreement statistics ($88.8\%$ adjacent agreement).
- 🔍 **Top 5 Failure Modes:** Real error traces and diagnostic hypotheses.
- 📝 **12 Non-Obvious Decision Logs:** Architectural trade-offs and rationale.

---

## 📊 Headline Evaluation Results (200 Golden Samples)

| Evaluation Metric | Trivial Baseline | Simple Baseline | **Proposed System (Grounded RAG + Triage)** |
| :--- | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 17.0% | 81.5% | **83.5%** |
| **Intent Macro F1-Score** | 0.042 | 0.766 | **0.795** |
| **Escalation Precision** | 100.0% | 50.0% | **53.8%** |
| **Escalation Recall** | 2.2% | 2.2% | **93.3%** |
| **Escalation F1-Score** | 0.043 | 0.043 | **0.683** |
| **ROUGE-L F1 (Lexical Match)** | 0.221 | 0.133 | **0.278** |
| **Official Link Compliance** | 100.0% | 41.5% | **76.5%** |
| **LLM-as-Judge Quality (/ 5.0)** | 4.29 | 3.88 | **4.32 / 5.0** |
| **Mean Runtime Latency** | 0.0 ms | 0.2 ms | **1.5 ms** |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Customer Inbound Tweet] --> B[Preprocessor & Normalizer]
    B --> C[Intent Classifier (7 Intents)]
    C -->|Intent + Confidence| D[Escalation & Triage Reasoner]
    B --> E[Historical Resolution RAG Index]
    E -->|Top-K Verified QA Pairs| F[Grounded Reply Generator]
    D -->|Escalation Decision + Stated Reason| G[Safety Arbiter & Policy Guardrails]
    F --> G
    G --> H[Final Output: Intent, Triage Decision, Reason, Grounded Reply]
```

### Core Components:
1. **7-Intent Classifier (`src/agent/intent_classifier.py`):** Calibrated multi-class model with margin-based confidence estimation.
2. **Historical Resolution RAG (`src/agent/retriever.py`):** Indexes 500 verified `@AmazonHelp` resolution pairs to ground draft replies without hallucination.
3. **Safety & Triage Engine (`src/agent/escalation_engine.py`):** High-recall ($93.3\%$) escalation engine routing account hacks, damaged goods, carrier disputes, and high-anger churn cases to human agents with an explicit stated reason code.
4. **Grounded Reply Generator (`src/agent/reply_generator.py`):** Drafts on-brand, empathetic replies constrained to $< 280$ characters with official self-service links (`amzn.to/...`) or secure DM routing (`amzn.to/AmazonDM`).

---

## 🎯 Golden Evaluation Set & Human Calibration

- **`data/golden_eval_set.json` (200 instances):** Hand-curated, stratified dataset across 7 intents and 5 difficulty tiers (`Standard`, `Noisy/Typos`, `High-Anger/Churn`, `Ambiguous/Edge-Case`, `Multi-Turn`). Zero leakage with the retrieval memory.
- **`data/human_judge_sample.json` (50 instances):** Double-scored by human annotators across 5 quality dimensions (Groundedness, Actionability, Brand Tone, Safety/Escalation, Conciseness) to validate the LLM Judge.
- **Inter-Rater Reliability:** **$88.8\%$ Adjacent Agreement ($\pm 1$ score)**, $96.0\%$ agreement on safety escalation decisions, proving the evaluation judge is reliable.

---

## 📑 Full Research Report

For the complete 6-page comprehensive report covering:
1. Problem framing & intentional non-goals
2. Tri-model experimental results & analysis
3. Top 5 failure modes with real snippets & hypotheses
4. **"What is misleading about my headline number?"** (Mandatory section on Goodhart's law & distribution shift)
5. Roadmap with 1 more week (tool calling, active learning)
6. 12-item Decision Log

👉 **Read the full report in [`REPORT.md`](file:///e:/Downloads/Hiver/REPORT.md)**

---

## 📂 Project Structure

```
├── data/
│   ├── raw_amazon_sample.json       # Historical resolution knowledge base (500 QA pairs)
│   ├── golden_eval_set.json         # 200 hand-curated & stratified golden test samples
│   ├── golden_eval_set.csv          # CSV format of golden evaluation set
│   └── human_judge_sample.json      # 50 human calibration instances
├── src/
│   ├── config.py                    # Intent taxonomy, escalation reasons, and thresholds
│   ├── data/
│   │   ├── dataset_loader.py        # Dataset loading routines
│   │   └── curator.py               # Dataset curation and serialization tools
│   ├── agent/
│   │   ├── intent_classifier.py     # Intent classification engine
│   │   ├── retriever.py             # Historical resolution RAG retriever
│   │   ├── escalation_engine.py     # Triage & decision reasoner
│   │   ├── reply_generator.py       # Grounded reply generator
│   │   └── pipeline.py              # Unified AmazonSupportAgent pipeline
│   ├── baselines/
│   │   ├── trivial_baseline.py      # Baseline 1: Majority class + canned response
│   │   └── simple_baseline.py       # Baseline 2: Naive Bayes + naive rules
│   └── eval/
│       ├── metrics.py               # Classification, triage, and ROUGE metrics
│       ├── llm_judge.py             # 5-dimension rubric LLM Judge
│       ├── human_agreement.py       # Statistical human vs judge calibration
│       └── benchmark_runner.py      # Tri-model evaluation harness
├── web/
│   ├── server.py                    # FastAPI server
│   └── static/
│       ├── index.html               # Sleek interactive dashboard
│       ├── style.css                # Dark mode & glassmorphic styling
│       └── app.js                   # Dashboard client logic
├── reproduce_results.py             # Single-command < 15-min reproduction script
├── run_demo.py                      # Interactive Web demo launcher
├── REPORT.md                        # Full technical research report
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## ⚖️ Rules & Citations
- **Dataset:** Primary dataset derived from *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`) hosted on Kaggle.
- **Dependencies:** Built using Python 3, scikit-learn, FastAPI, uvicorn, rich, and rouge-score. Optional cloud LLM integration supports OpenAI, Google Gemini, and Anthropic APIs via `.env`.
