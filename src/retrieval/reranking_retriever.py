from pathlib import Path
from urllib.parse import urlparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml

from src.index.faiss_searcher import FaissSpanSearcher


class RerankingSpanRetriever:
    """
    Wraps:
      - FaissSpanSearcher (fast dense retrieval)
      - Cross-encoder reranker (MPNet) trained in 07

    Usage:
        retr = RerankingSpanRetriever(PROJECT_ROOT, domain="money-heist")
        results = retr.search("Who is Raquel Murillo?", faiss_k=50, top_k=5)
    """

    def __init__(
        self,
        project_root: Path,
        domain: str | None = None,
        model_subdir: str = "models/reranker",
        device: str | None = None,
        verbose: bool = False,
    ):
        self.project_root = Path(project_root)

        # --- resolve domain from scraping.yaml if not given ---
        if domain is None:
            cfg_path = self.project_root / "configs" / "scraping.yaml"
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            base_url = cfg["base_url"].rstrip("/")
            domain = urlparse(base_url).netloc.split(".")[0]
        self.domain = domain

        # --- device ---
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # --- FAISS searcher ---
        self.faiss = FaissSpanSearcher(self.project_root, domain=self.domain, verbose=verbose)

        # --- reranker model + tokenizer ---
        self.model_dir = self.project_root / model_subdir / self.domain / "best"
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Reranker model directory not found: {self.model_dir}")

        if verbose:
            print(f"[RerankingSpanRetriever] Loading reranker from {self.model_dir} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Given a query and FAISS candidate spans, compute cross-encoder scores
        and return the same list sorted by reranker score (desc).
        """
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        batch = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=320,
            return_tensors="pt",
        )

        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits.squeeze(-1)  # (N,)
        scores = logits.cpu().tolist()

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # sort by rerank_score desc
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return candidates

    def search(
        self,
        query: str,
        faiss_k: int = 50,
        top_k: int = 10,
    ) -> list[dict]:
        """
        1) Use FAISS to get faiss_k candidates
        2) Rerank with cross-encoder
        3) Return top_k
        """
        faiss_results = self.faiss.search(query, top_k=faiss_k)
        if not faiss_results:
            return []

        reranked = self._rerank(query, faiss_results)
        return reranked[:top_k]