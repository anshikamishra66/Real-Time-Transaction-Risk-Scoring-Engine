"""FastAPI serving layer for the Real-Time Transaction Risk Scoring Engine."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db, scoring
from api.schemas import ScoreResponse, TransactionRequest
from models.config import METRICS_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    scoring.load_models()
    yield


app = FastAPI(
    title="Real-Time Transaction Risk Scoring Engine",
    description="AI-powered fraud detection and compliance automation for payment transactions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score-transaction", response_model=ScoreResponse)
def score_transaction(request: TransactionRequest) -> ScoreResponse:
    try:
        return scoring.score_transaction(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/transactions/recent")
def recent_transactions(limit: int = 50) -> list[dict]:
    return db.fetch_recent(limit=limit)


@app.get("/review-queue")
def review_queue(status: Optional[str] = "pending", limit: int = 100) -> list[dict]:
    return db.fetch_review_queue(status=status, limit=limit)


@app.post("/review-queue/{transaction_id}/resolve")
def resolve_review(transaction_id: str, status: str = "resolved") -> dict:
    db.update_review_status(transaction_id, status)
    return {"transaction_id": transaction_id, "review_status": status}


@app.get("/metrics")
def model_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not found -- run models/train_all.py first.")
    return json.loads(METRICS_PATH.read_text())


@app.post("/admin/reset-demo")
def reset_demo() -> dict:
    """Clears the transaction log and in-memory feature-store state -- lets a
    demo be replayed from a clean slate without restarting the process."""
    db.reset_db()
    scoring.feature_store.reset()
    return {"status": "reset"}
