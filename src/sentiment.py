import os

MODEL_NAME = os.getenv(
    "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
)


class SentimentAnalyzer:
    """Thin wrapper around a Hugging Face sentiment pipeline.

    The model loads lazily on first use so importing this module stays cheap.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._pipe = None

    def load(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("sentiment-analysis", model=self.model_name)
        return self._pipe

    def analyze(self, text: str) -> dict:
        result = self.load()(text)[0]
        return {"label": result["label"], "score": round(float(result["score"]), 4)}

    def analyze_batch(self, texts: list) -> list:
        results = self.load()(list(texts))
        return [
            {"label": r["label"], "score": round(float(r["score"]), 4)}
            for r in results
        ]


_analyzer = None


def get_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
