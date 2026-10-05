# sentiment-api

A small REST API for sentiment analysis. Send it text, get back `POSITIVE` or `NEGATIVE` with a confidence score. I built it as a clean starting point for anything that needs text classification behind an HTTP endpoint — review dashboards, feedback triage, comment moderation, that sort of thing.

It uses a fine-tuned DistilBERT model from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`), served with FastAPI.

## Quickstart

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate

# CPU-only torch — much smaller download than the CUDA build, plenty for inference
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn src.app:app --reload
```

The API is at `http://localhost:8000`, interactive docs at `http://localhost:8000/docs`.

Heads up: the first run downloads the model weights (~260MB) from Hugging Face and caches them locally, so the first request takes a minute. After that it's fast.

## Try it

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The onboarding was smooth and the team was great"}'
# {"label":"POSITIVE","score":0.9998}

curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["loved it", "worst experience ever", "it was okay"]}'
# {"count":3,"results":[{"label":"POSITIVE",...},{"label":"NEGATIVE",...},{"label":"POSITIVE",...}]}
```

## Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## Endpoints

| Method | Path             | Description                                        |
|--------|------------------|----------------------------------------------------|
| GET    | `/health`        | Liveness check                                     |
| GET    | `/model-info`    | Model name, task, and label set                    |
| POST   | `/analyze`       | Analyze one text — `{"text": "..."}`               |
| POST   | `/analyze/batch` | Analyze up to 50 texts — `{"texts": [...]}`        |

Guards: batch requests are capped at 50 texts, each text at 2000 characters, and there's a simple in-memory rate limit (100 requests/minute per IP, configurable). These are meant as sane defaults for a small service, not production-grade protection.

## Project structure

```
sentiment-api/
├── src/
│   ├── app.py        # FastAPI app, routes, validation, rate limiting
│   └── sentiment.py  # SentimentAnalyzer — lazy-loaded HF pipeline
├── tests/
│   └── test_api.py   # TestClient tests (model is stubbed out)
├── Dockerfile
├── requirements.txt
├── .env.example
└── LICENSE
```

## Configuration

Copy `.env.example` to `.env` to override defaults:

- `MODEL_NAME` — any Hugging Face sentiment model id
- `MAX_BATCH_SIZE` — max texts per batch request (default 50)
- `MAX_TEXT_LENGTH` — max characters per text (default 2000)
- `RATE_LIMIT_PER_MINUTE` — per-IP request cap (default 100)

## Running tests

```bash
pytest
```

Tests stub out the model so they run offline in seconds.

## Tech stack

Python 3.11, FastAPI, Pydantic v2, Hugging Face Transformers, PyTorch (CPU), uvicorn, Docker.

## License

MIT — see [LICENSE](LICENSE).
