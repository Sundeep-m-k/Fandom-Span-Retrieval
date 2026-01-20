from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import yaml
from urllib.parse import urlparse


class FaissSpanSearcher:
    """
    Thin wrapper around:
      - spans_{domain}.csv
      - spans_{domain}.npy
      - spans_{domain}.index_ids.npy
      - faiss_flat_{domain}.index

    Provides:
      - search(query, top_k=10) -> List[dict]
    """

    def __init__(
        self,
        project_root: Path,
        domain: str | None = None,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.project_root = Path(project_root)
        self.verbose = verbose

        # Resolve domain from scraping.yaml if not provided
        if domain is None:
            cfg_path = self.project_root / "configs" / "scraping.yaml"
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            base_url = cfg["base_url"].rstrip("/")
            domain = urlparse(base_url).netloc.split(".")[0]

        self.domain = domain

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Paths
        emb_dir = self.project_root / "data" / "embeddings"
        idx_dir = self.project_root / "data" / "indexes"
        proc_dir = self.project_root / "data" / "processed"

        self.emb_path = emb_dir / f"spans_{domain}.npy"
        self.ids_path = emb_dir / f"spans_{domain}.index_ids.npy"
        self.index_path = idx_dir / f"faiss_flat_{domain}.index"
        self.spans_csv = proc_dir / f"spans_{domain}.csv"

        if self.verbose:
            print("=== FaissSpanSearcher init ===")
            print(f"PROJECT_ROOT: {self.project_root}")
            print(f"DOMAIN:       {self.domain}")
            print(f"Embeddings:   {self.emb_path}")
            print(f"Index IDs:    {self.ids_path}")
            print(f"FAISS index:  {self.index_path}")
            print(f"Spans CSV:    {self.spans_csv}")

        # Load everything
        self._load_resources()

    def _load_resources(self):
        # Load embeddings just for shape check (not strictly needed to search)
        if not self.emb_path.exists():
            raise FileNotFoundError(self.emb_path)
        self.embeddings = np.load(self.emb_path).astype("float32")
        if self.verbose:
            print(f"Loaded embeddings shape: {self.embeddings.shape}")

        # Index IDs
        if not self.ids_path.exists():
            raise FileNotFoundError(self.ids_path)
        self.span_ids = np.load(self.ids_path, allow_pickle=True).astype(str)
        if self.verbose:
            print(f"Loaded {len(self.span_ids)} span_ids")

        # Spans dataframe
        if not self.spans_csv.exists():
            raise FileNotFoundError(self.spans_csv)
        self.df_spans = pd.read_csv(self.spans_csv)
        self.df_spans = self.df_spans.set_index("span_id", drop=False)
        if self.verbose:
            print(f"Spans DF rows: {len(self.df_spans)}")

        # FAISS index
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        if self.verbose:
            print(f"FAISS index ntotal: {self.index.ntotal}")

        # SentenceTransformer encoder for queries
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if self.verbose:
            print(f"Loading SentenceTransformer model on {self.device}: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=self.device)

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query to a FAISS-compatible vector.
        Assumes the index was built with L2-normalized embeddings and IndexFlatIP.
        """
        emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the FAISS index with the given query string.

        Returns:
            List of dicts with:
                - span_id
                - score (similarity)
                - text
                - article_id, page_name, title, section, url, start_char, end_char
        """
        if top_k <= 0:
            return []

        q_vec = self._encode_query(query)  # shape (1, d)
        scores, idxs = self.index.search(q_vec, top_k)
        scores = scores[0]
        idxs = idxs[0]

        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            if idx >= len(self.span_ids):
                continue
            span_id = self.span_ids[idx]
            # df_spans index is span_id
            if span_id not in self.df_spans.index:
                continue
            row = self.df_spans.loc[span_id]
            results.append(
                {
                    "span_id": span_id,
                    "score": float(score),
                    "text": row["text"],
                    "article_id": int(row["article_id"]),
                    "page_name": row["page_name"],
                    "title": row["title"],
                    "section": row["section"],
                    "url": row["url"],
                    "start_char": int(row["start_char"]),
                    "end_char": int(row["end_char"]),
                }
            )
        return results