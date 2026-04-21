"""
retrieval_eval.py - Computes retrieval Hit Rate and MRR from agent responses.

Workflow:
  1. For each test case, call agent.query(question) to get actual retrieved_ids.
  2. Compare retrieved_ids against ground_truth_retrieval_ids.
  3. Compute Hit Rate (binary) and MRR (position-aware) per case.
  4. Return aggregate metrics and per-case results for later correlation analysis.
"""

import asyncio
from typing import Dict, List

from config import RETRIEVAL_EVAL_CONCURRENCY


class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 5,
    ) -> float:
        """
        Hit Rate = 1.0 if at least one expected id appears in top_k retrieved ids.
        Hit Rate = 0.0 otherwise.
        """
        if not expected_ids:
            return None  # Adversarial case - no ground truth, skip from retrieval metrics

        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR = 1 / rank of the first expected id found (1-indexed).
        If none found -> 0.0.
        """
        if not expected_ids:
            return None

        for index, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return round(1.0 / (index + 1), 4)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict], agent) -> Dict:
        """
        Run retrieval evaluation for the full dataset concurrently.

        Args:
            dataset: list of test cases from golden_set.jsonl
            agent: Agent instance exposing async query(question) -> Dict with retrieved_ids
        """
        semaphore = asyncio.Semaphore(RETRIEVAL_EVAL_CONCURRENCY)

        async def eval_single(case: Dict) -> Dict:
            question = case.get("question", "")
            expected_ids = case.get("ground_truth_retrieval_ids", [])

            async with semaphore:
                try:
                    response = await agent.query(question)
                    retrieved_ids = response.get("retrieved_ids", [])
                except Exception:
                    retrieved_ids = []

            hit = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=5)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)
            return {
                "question": question[:80],
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids[:5],
                "hit_rate": hit,
                "mrr": mrr,
                "difficulty": case.get("metadata", {}).get("difficulty", "unknown"),
                "type": case.get("metadata", {}).get("type", "unknown"),
            }

        per_case_results = await asyncio.gather(*[eval_single(case) for case in dataset])

        hit_rates = [result["hit_rate"] for result in per_case_results if result["hit_rate"] is not None]
        mrrs = [result["mrr"] for result in per_case_results if result["mrr"] is not None]
        skipped_adversarial = sum(1 for result in per_case_results if result["hit_rate"] is None)

        evaluated = len(hit_rates)
        avg_hit_rate = round(sum(hit_rates) / evaluated, 4) if evaluated else 0.0
        avg_mrr = round(sum(mrrs) / evaluated, 4) if evaluated else 0.0

        difficulty_breakdown = {}
        for result in per_case_results:
            if result["hit_rate"] is None:
                continue

            difficulty = result["difficulty"]
            if difficulty not in difficulty_breakdown:
                difficulty_breakdown[difficulty] = {"hit_rates": [], "mrrs": []}

            difficulty_breakdown[difficulty]["hit_rates"].append(result["hit_rate"])
            difficulty_breakdown[difficulty]["mrrs"].append(result["mrr"])

        difficulty_summary = {}
        for difficulty, values in difficulty_breakdown.items():
            count = len(values["hit_rates"])
            difficulty_summary[difficulty] = {
                "count": count,
                "avg_hit_rate": round(sum(values["hit_rates"]) / count, 4) if count else 0,
                "avg_mrr": round(sum(values["mrrs"]) / count, 4) if count else 0,
            }

        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "total_evaluated": evaluated,
            "skipped_adversarial": skipped_adversarial,
            "perfect_hit_count": sum(1 for hit in hit_rates if hit == 1.0),
            "zero_hit_count": sum(1 for hit in hit_rates if hit == 0.0),
            "difficulty_breakdown": difficulty_summary,
            "per_case": per_case_results,
        }
