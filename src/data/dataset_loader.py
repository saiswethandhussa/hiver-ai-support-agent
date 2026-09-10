"""
Dataset Loader for @AmazonHelp Customer Support Data.
Provides clean APIs to load knowledge base, golden eval set, and human calibration sample.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import DATA_DIR
from src.data.curator import build_datasets

def load_knowledge_base() -> List[Dict[str, Any]]:
    """Loads the historical resolution knowledge base (500 QA pairs)."""
    kb_path = DATA_DIR / "raw_amazon_sample.json"
    if not kb_path.exists():
        build_datasets()
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_golden_eval_set() -> List[Dict[str, Any]]:
    """Loads the 200 hand-curated and stratified golden evaluation set."""
    golden_path = DATA_DIR / "golden_eval_set.json"
    if not golden_path.exists():
        build_datasets()
    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_human_judge_sample() -> List[Dict[str, Any]]:
    """Loads the 50 human-annotated rubric benchmark samples."""
    hj_path = DATA_DIR / "human_judge_sample.json"
    if not hj_path.exists():
        build_datasets()
    with open(hj_path, "r", encoding="utf-8") as f:
        return json.load(f)
