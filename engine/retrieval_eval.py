"""
Retrieval Evaluator – Day 14 AI Evaluation
Calculates Hit Rate and MRR for the Vector DB retrieval stage.
"""
from typing import List, Dict


class RetrievalEvaluator:
    """Evaluates the retrieval quality of the RAG pipeline."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    # ── Core Metrics ─────────────────────────────────────────────────────────

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int | None = None,
    ) -> float:
        """
        Hit Rate @k: 1.0 if at least one expected_id appears in the top_k
        retrieved chunks, otherwise 0.0.
        """
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
    ) -> float:
        """
        Mean Reciprocal Rank (MRR):
        Finds the rank of the FIRST matching expected_id in retrieved_ids.
        MRR = 1 / rank (1-indexed). Returns 0 if no match found.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    # ── Per-case scorer (used by engine/runner.py) ────────────────────────────

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        """
        Compute retrieval metrics for a single test case.
        test_case must have 'expected_retrieval_ids'.
        response must have 'retrieved_ids'.
        Returns dict with faithfulness, relevancy, and retrieval sub-scores.
        """
        expected_ids: List[str] = test_case.get("expected_retrieval_ids", [])
        retrieved_ids: List[str] = response.get("retrieved_ids", [])

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        # Simple faithfulness proxy: hit_rate (context was retrieved)
        # Simple relevancy proxy: normalized MRR (higher rank = more relevant)
        faithfulness = hit_rate
        relevancy = mrr

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids[:self.top_k],
            },
        }

    # ── Batch evaluation ─────────────────────────────────────────────────────

    async def evaluate_batch(self, dataset: List[Dict], responses: List[Dict]) -> Dict:
        """
        Run retrieval eval for an entire dataset.
        Returns average hit_rate and mrr across all cases.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        hit_rates, mrrs = [], []
        for case, resp in zip(dataset, responses):
            result = await self.score(case, resp)
            hit_rates.append(result["retrieval"]["hit_rate"])
            mrrs.append(result["retrieval"]["mrr"])

        return {
            "avg_hit_rate": sum(hit_rates) / len(hit_rates),
            "avg_mrr": sum(mrrs) / len(mrrs),
        }
