import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .sentiment import get_analyzer

load_dotenv()

MAX_BATCH = int(os.getenv("MAX_BATCH_SIZE", "50"))
MAX_CHARS = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
WINDOW_SECONDS = 60

# naive in-memory rate limiting: timestamps of recent requests per client IP
_hits: dict = defaultdict(list)


def check_rate_limit(ip: str):
    now = time.time()
    bucket = _hits[ip]
    bucket[:] = [t for t in bucket if now - t < WINDOW_SECONDS]
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded, try again in a minute."
        )
    bucket.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm the model at startup so the first real request isn't slow
    get_analyzer().load()
    yield


app = FastAPI(title="Sentiment API", version="0.1.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def no_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: list = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def no_blank_entries(cls, v: list) -> list:
        cleaned = [t.strip() for t in v]
        if any(not t for t in cleaned):
            raise ValueError("texts must not contain blank entries")
        if any(len(t) > MAX_CHARS for t in cleaned):
            raise ValueError(f"each text must be {MAX_CHARS} characters or fewer")
        return cleaned


class SentimentResult(BaseModel):
    label: str
    score: float


class BatchResponse(BaseModel):
    count: int
    results: list


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": get_analyzer().model_name,
        "task": "sentiment-analysis",
        "labels": ["POSITIVE", "NEGATIVE"],
    }


@app.post("/analyze", response_model=SentimentResult)
def analyze(payload: AnalyzeRequest, request: Request):
    check_rate_limit(_client_ip(request))
    return get_analyzer().analyze(payload.text)


@app.post("/analyze/batch", response_model=BatchResponse)
def analyze_batch(payload: BatchRequest, request: Request):
    check_rate_limit(_client_ip(request))
    results = get_analyzer().analyze_batch(payload.texts)
    return {"count": len(results), "results": results}
