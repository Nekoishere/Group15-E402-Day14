import os
from typing import Any

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

from config import CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_K

load_dotenv()


class VectorStore:
    """ChromaDB-backed vector store for academic regulation chunks."""

    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Public API ────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add chunks to the vector store.
        Each chunk: {"text": str, "metadata": dict, "id": str}
        Embeds in batches of 100 to stay within OpenAI limits.
        """
        if not chunks:
            return

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            embeddings = self._embed(texts)
            self._collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

    def query(self, query_text: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed query and retrieve top_k most similar chunks.
        Returns list of {"text", "metadata", "distance"} dicts.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self._embed([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            chunks.append({
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })
        return chunks

    def delete_by_pdf_id(self, pdf_id: str) -> int:
        """Delete all chunks belonging to a PDF. Returns count deleted."""
        try:
            # Get IDs first
            results = self._collection.get(
                where={"pdf_id": pdf_id},
                include=["metadatas"],
            )
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def count(self) -> int:
        return self._collection.count()

    # ── Internal ──────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API for a list of texts."""
        # Clean texts: replace null bytes and excessive whitespace
        cleaned = [t.replace("\x00", " ").strip() or " " for t in texts]
        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=cleaned,
        )
        return [item.embedding for item in response.data]
