"""HuggingFace model manager: lazy loading, CUDA auto-detect, idle unload."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "bio_clinical_bert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmed_bert": "NLP4Science/pubmedbert-full-text-clinical",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_large": "BAAI/bge-large-en-v1.5",
    "bge_reranker": "BAAI/bge-reranker-large",
    "bart_cnn": "facebook/bart-large-cnn",
}

IDLE_TIMEOUT_SECONDS = 600


class HFModelManager:
    _instance: Optional["HFModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, models_dir: str = "models"):
        if self._initialized:
            return
        self._initialized = True
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict = {}
        self._timers: dict[str, threading.Timer] = {}
        self._device = self._detect_device()
        self._knowledge_entries: list[dict] = []
        self._knowledge_index = None
        logger.info("HFModelManager initialized on device=%s", self._device)

    def _detect_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _reset_timer(self, key: str) -> None:
        if key in self._timers:
            self._timers[key].cancel()
        t = threading.Timer(IDLE_TIMEOUT_SECONDS, self._unload, args=[key])
        t.daemon = True
        t.start()
        self._timers[key] = t

    def _unload(self, key: str) -> None:
        if key in self._loaded:
            del self._loaded[key]
            logger.info("Unloaded idle model: %s", key)

    def _load_sentence_transformer(self, key: str) -> object:
        if key not in self._loaded:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_REGISTRY[key], cache_folder=str(self._models_dir))
                if self._device == "cuda":
                    model = model.to("cuda")
                self._loaded[key] = model
            except ImportError:
                logger.warning("sentence-transformers not installed — using TF-IDF fallback")
                self._loaded[key] = None
        self._reset_timer(key)
        return self._loaded[key]

    def _load_cross_encoder(self, key: str) -> object:
        if key not in self._loaded:
            try:
                from sentence_transformers import CrossEncoder
                self._loaded[key] = CrossEncoder(MODEL_REGISTRY[key], max_length=512)
            except ImportError:
                logger.warning("sentence-transformers not installed — using keyword reranker")
                self._loaded[key] = None
        self._reset_timer(key)
        return self._loaded[key]

    def _load_pipeline(self, key: str, task: str) -> object:
        if key not in self._loaded:
            try:
                from transformers import pipeline
                self._loaded[key] = pipeline(
                    task,
                    model=MODEL_REGISTRY[key],
                    device=0 if self._device == "cuda" else -1,
                )
            except ImportError:
                logger.warning("transformers not installed — using heuristic fallback for %s", key)
                self._loaded[key] = None
        self._reset_timer(key)
        return self._loaded[key]

    def extract_medical_entities(self, text: str) -> list[str]:
        """Extract medical entities using Bio_ClinicalBERT NER."""
        pipeline = self._load_pipeline("bio_clinical_bert", "ner")
        if pipeline is None:
            return self._heuristic_ner(text)
        try:
            results = pipeline(text[:512], aggregation_strategy="simple")
            entities = []
            for r in results:
                if r.get("score", 0) > 0.7:
                    entities.append(r.get("word", "").replace("##", ""))
            return list(set(entities)) if entities else self._heuristic_ner(text)
        except Exception as e:
            logger.warning("Bio_ClinicalBERT NER failed: %s", e)
            return self._heuristic_ner(text)

    def _heuristic_ner(self, text: str) -> list[str]:
        medical_terms = [
            "pain", "fever", "cough", "nausea", "headache", "dizziness",
            "fatigue", "weakness", "shortness of breath", "chest pain",
            "abdominal pain", "vomiting", "diarrhea", "swelling", "rash",
            "palpitations", "confusion", "seizure", "bleeding",
        ]
        text_lower = text.lower()
        return [term for term in medical_terms if term in text_lower]

    def classify_clinical_text(self, text: str, labels: Optional[list[str]] = None) -> dict:
        """Classify clinical text using PubMedBERT."""
        default_labels = [
            "cardiovascular disease", "respiratory disease", "neurological disorder",
            "gastrointestinal disorder", "infectious disease", "musculoskeletal disorder",
        ]
        labels = labels or default_labels
        try:
            from transformers import pipeline as hf_pipeline
            classifier = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self._device == "cuda" else -1,
            )
            result = classifier(text[:512], candidate_labels=labels)
            return {label: score for label, score in zip(result["labels"], result["scores"])}
        except Exception as e:
            logger.warning("Classification failed: %s — using keyword heuristic", e)
            return self._heuristic_classify(text, labels)

    def _heuristic_classify(self, text: str, labels: list[str]) -> dict:
        keyword_map = {
            "cardiovascular": ["chest", "heart", "palpitation", "angina"],
            "respiratory": ["cough", "breath", "wheeze", "lung"],
            "neurological": ["headache", "dizzy", "vision", "speech"],
            "gastrointestinal": ["stomach", "nausea", "vomit", "diarrhea"],
            "infectious": ["fever", "infection", "bacteria", "virus"],
        }
        text_lower = text.lower()
        scores = {}
        for label in labels:
            score = 0.1
            for category, keywords in keyword_map.items():
                if category in label.lower() and any(kw in text_lower for kw in keywords):
                    score = 0.8
                    break
            scores[label] = score
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def encode(self, text: str, key: str = "bge_large") -> list[float]:
        model = self._load_sentence_transformer(key)
        if model is None:
            return self._tfidf_fallback_encode(text)
        try:
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning("Encode failed: %s", e)
            return self._tfidf_fallback_encode(text)

    def encode_batch(self, texts: list[str], key: str = "bge_large") -> list[list[float]]:
        model = self._load_sentence_transformer(key)
        if model is None:
            return [self._tfidf_fallback_encode(t) for t in texts]
        try:
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
            return embeddings.tolist()
        except Exception as e:
            logger.warning("Batch encode failed: %s", e)
            return [self._tfidf_fallback_encode(t) for t in texts]

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        model = self._load_cross_encoder("bge_reranker")
        if model is None:
            return self._heuristic_rerank(query, documents, top_k)
        try:
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(pairs)
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return indexed[:top_k]
        except Exception as e:
            logger.warning("Rerank failed: %s", e)
            return self._heuristic_rerank(query, documents, top_k)

    def summarize(self, text: str, max_length: int = 150) -> str:
        pipe = self._load_pipeline("bart_cnn", "summarization")
        if pipe is None:
            return self._extractive_fallback(text, max_length)
        try:
            result = pipe(text[:1024], max_length=max_length, min_length=30, do_sample=False)
            return result[0]["summary_text"]
        except Exception as e:
            logger.warning("Summarization failed: %s", e)
            return self._extractive_fallback(text, max_length)

    def retrieve_from_knowledge_brain(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant entries from SECOND-KNOWLEDGE-BRAIN.md."""
        if not self._knowledge_entries:
            self._load_knowledge_brain()
        if not self._knowledge_entries:
            return []

        query_emb = self.encode(query, key="bge_large")
        doc_texts = [
            f"{e.get('title', '')} {e.get('abstract', '')} {e.get('key_finding', '')}"
            for e in self._knowledge_entries
        ]
        doc_embs = self.encode_batch(doc_texts[:50], key="bge_large")

        import numpy as np
        q = np.array(query_emb)
        D = np.array(doc_embs)
        scores = (D @ q).tolist()
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self._knowledge_entries[i] for i, _ in indexed[:top_k]]

    def _load_knowledge_brain(self) -> None:
        from pathlib import Path
        brain_path = Path("SECOND-KNOWLEDGE-BRAIN.md")
        if not brain_path.exists():
            return
        content = brain_path.read_text(encoding="utf-8")
        entries = []
        in_table = False
        header_passed = False
        for line in content.split("\n"):
            if "| Title |" in line and "Authors" in line:
                in_table = True
                header_passed = False
                continue
            if in_table and line.startswith("|---"):
                header_passed = True
                continue
            if in_table and header_passed and line.startswith("|") and "|" in line[1:]:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 5:
                    entries.append({
                        "title": parts[0] if parts else "",
                        "authors": parts[1] if len(parts) > 1 else "",
                        "year": parts[2] if len(parts) > 2 else "",
                        "venue": parts[3] if len(parts) > 3 else "",
                        "url": parts[4] if len(parts) > 4 else "",
                        "key_finding": parts[5] if len(parts) > 5 else "",
                        "abstract": parts[5] if len(parts) > 5 else "",
                    })
            elif in_table and not line.startswith("|"):
                in_table = False
        self._knowledge_entries = entries
        logger.info("Loaded %d entries from SECOND-KNOWLEDGE-BRAIN.md", len(entries))

    def _tfidf_fallback_encode(self, text: str, dim: int = 256) -> list[float]:
        import hashlib
        import math
        tokens = text.lower().split()
        vec = [0.0] * dim
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % dim
            tfidf = 1.0 + math.log(1 + tokens.count(token))
            vec[idx] += tfidf
        magnitude = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / magnitude for x in vec]

    def _heuristic_rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        query_words = set(query.lower().split())
        scores = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words) / (len(query_words) + 1)
            scores.append((i, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def _extractive_fallback(self, text: str, max_length: int) -> str:
        sentences = text.split(". ")
        result = ""
        for s in sentences:
            if len(result) + len(s) < max_length:
                result += s + ". "
            else:
                break
        return result.strip() or text[:max_length]
