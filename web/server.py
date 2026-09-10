"""
FastAPI Server & REST API for @AmazonHelp AI Customer Support Agent.
Provides interactive live testing, batch benchmark metrics, and human-judge calibration data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import json

from src.config import INTENT_TAXONOMY, ESCALATION_REASONS, ARTIFACTS_DIR
from src.agent.pipeline import AmazonSupportAgent
from src.data.dataset_loader import load_golden_eval_set, load_human_judge_sample
from src.eval.benchmark_runner import run_full_benchmark

app = FastAPI(title="Amazon Support AI Agent API", version="1.0.0")

# Initialize agent once on startup
agent = AmazonSupportAgent()

class TweetRequest(BaseModel):
    customer_text: str

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "brand": "@AmazonHelp", "agent_initialized": True}

@app.get("/api/config")
def get_config():
    return {
        "brand": "@AmazonHelp",
        "intents": INTENT_TAXONOMY,
        "escalation_reasons": ESCALATION_REASONS
    }

@app.post("/api/predict")
def predict_tweet(req: TweetRequest):
    if not req.customer_text or len(req.customer_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Customer text cannot be empty.")
    
    response = agent.process_tweet(req.customer_text)
    return response.model_dump()

@app.get("/api/benchmark")
def get_benchmark_results():
    artifact_path = ARTIFACTS_DIR / "benchmark_results.json"
    if artifact_path.exists():
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return run_full_benchmark()

@app.post("/api/benchmark/run")
def trigger_benchmark():
    return run_full_benchmark()

@app.get("/api/golden_set")
def get_golden_set():
    return load_golden_eval_set()

@app.get("/api/human_agreement")
def get_human_agreement():
    artifact_path = ARTIFACTS_DIR / "benchmark_results.json"
    if artifact_path.exists():
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("human_judge_agreement", {})
    return {}

# Mount static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
