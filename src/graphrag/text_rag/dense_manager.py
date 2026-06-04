from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.embeddings import Embeddings

from graphrag.text_rag.manager import TextChunk

logger = logging.getLogger("graphrag")


class _PrefixedEmbeddings(Embeddings):
    """LangChain Embeddings that prepends query/passage prefixes.

    Required for multilingual-e5-style models. Prefix skipped when empty string.
    """

    def __init__(
        self,
        inner: Embeddings,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        self._inner = inner
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._passage_prefix:
            texts = [f"{self._passage_prefix}{t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        return self._inner.embed_query(text)


def _build_embeddings(
    model_name: str,
    query_prefix: str,
    passage_prefix: str,
    normalize: bool,
    device: str,
) -> _PrefixedEmbeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    resolved_device = device
    if device == "auto":
        try:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    inner = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": resolved_device},
        encode_kwargs={"normalize_embeddings": normalize},
    )
    return _PrefixedEmbeddings(inner, query_prefix=query_prefix, passage_prefix=passage_prefix)


def _corpus_fingerprint(model_name: str, chunks: list[TextChunk]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.content.encode())
    return h.hexdigest()[:16]


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "-").replace(":", "-")


class DenseTextRAGManager:
    """FAISS-backed vector store manager with cosine similarity retrieval.

    Drop-in replacement for TextRAGManager — identical public interface.
    Embeddings use ``intfloat/multilingual-e5-base`` by default, which works
    for both English and Italian queries with ``query: ``/``passage: `` prefixes.

    The FAISS index is persisted to ``vector_index_dir`` keyed by a fingerprint
    of the model name and chunk contents; subsequent runs with the same corpus
    skip re-encoding and load from cache.
    """

    def __init__(
        self,
        embedding_model: str = "intfloat/multilingual-e5-base",
        vector_index_dir: str = "artifacts/vector_index",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        normalize: bool = True,
        device: str = "auto",
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_index_dir = Path(vector_index_dir)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._normalize = normalize
        self._device = device
        self._chunks: list[TextChunk] = []
        self._store = None  # FAISS | None
        self._embeddings: _PrefixedEmbeddings | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._store = None

    def _get_embeddings(self) -> _PrefixedEmbeddings:
        if self._embeddings is None:
            self._embeddings = _build_embeddings(
                model_name=self._embedding_model,
                query_prefix=self._query_prefix,
                passage_prefix=self._passage_prefix,
                normalize=self._normalize,
                device=self._device,
            )
        return self._embeddings

    def add_chunks(self, chunks: Iterable[TextChunk]) -> int:
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        chunk_list = [c for c in chunks if c.content.strip()]
        if not chunk_list:
            return 0

        self._chunks.extend(chunk_list)
        fingerprint = _corpus_fingerprint(self._embedding_model, self._chunks)
        cache_dir = self._vector_index_dir / f"{_model_slug(self._embedding_model)}-{fingerprint}"
        embeddings = self._get_embeddings()

        if cache_dir.exists():
            logger.info("DenseTextRAGManager: loading index from cache %s", cache_dir)
            self._store = FAISS.load_local(
                str(cache_dir),
                embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            logger.info(
                "DenseTextRAGManager: building FAISS index for %d chunks", len(self._chunks)
            )
            texts = [c.content for c in self._chunks]
            metadatas = [
                {"chunk_id": c.chunk_id, "source": c.source or ""}
                for c in self._chunks
            ]
            self._store = FAISS.from_texts(
                texts,
                embeddings,
                metadatas=metadatas,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(cache_dir))
            logger.info("DenseTextRAGManager: index saved to %s", cache_dir)

        return len(chunk_list)

    def retrieve_with_scores(
        self, query: str, top_k: int = 5
    ) -> list[tuple[TextChunk, float]]:
        if self._store is None or top_k <= 0:
            return []

        hits = self._store.similarity_search_with_score(query, k=top_k)
        # MAX_INNER_PRODUCT: score is inner product (== cosine for normalised embs),
        # higher means more similar. Results already ordered descending by FAISS.
        result: list[tuple[TextChunk, float]] = []
        for doc, score in hits:
            meta = doc.metadata
            chunk = TextChunk(
                chunk_id=meta.get("chunk_id", ""),
                content=doc.page_content,
                source=meta.get("source") or None,
            )
            result.append((chunk, float(score)))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def retrieve(self, query: str, top_k: int = 5) -> list[TextChunk]:
        return [chunk for chunk, _ in self.retrieve_with_scores(query=query, top_k=top_k)]

    def build_context(
        self, query: str, top_k: int = 4, separator: str = "\n\n---\n\n"
    ) -> str:
        chunks = self.retrieve(query=query, top_k=top_k)
        return separator.join(chunk.content for chunk in chunks)
