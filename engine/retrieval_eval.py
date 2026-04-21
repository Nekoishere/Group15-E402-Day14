"""
retrieval_eval.py — Tính toán Hit Rate & MRR thật từ agent responses.

Workflow:
  1. Với mỗi test case trong dataset, gọi agent.query(question) để lấy retrieved_ids thật.
  2. So sánh retrieved_ids với ground_truth_retrieval_ids.
  3. Tính Hit Rate (binary) và MRR (position-aware) cho từng case.
  4. Trả về tổng hợp + per-case results để phân tích correlation với Answer Score.
"""

import asyncio
from typing import List, Dict



class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 5) -> float:
        """
        Hit Rate = 1.0 nếu ít nhất 1 expected_id xuất hiện trong top_k retrieved_ids.
        Hit Rate = 0.0 nếu không có.
        """
        if not expected_ids:
            return None  # Adversarial case — không có ground truth, bỏ qua
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
    ) -> float:
        """
        MRR = 1 / rank của expected_id đầu tiên tìm thấy (1-indexed).
        Nếu không thấy → 0.0.
        Ví dụ: expected ở vị trí 1 → MRR=1.0, vị trí 2 → 0.5, vị trí 5 → 0.2
        """
        if not expected_ids:
            return None  # Adversarial case
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return round(1.0 / (i + 1), 4)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict], agent) -> Dict:
        """
        Chạy Retrieval Eval cho toàn bộ dataset:
        - Gọi agent.query() để lấy retrieved_ids thật
        - Tính Hit Rate & MRR từng case
        - Trả về aggregate metrics + per_case results để phân tích correlation

        Args:
            dataset: list of test cases (từ golden_set.jsonl)
            agent: Agent instance có method async query(question) -> Dict với key 'retrieved_ids'
        """
        semaphore = asyncio.Semaphore(10)  # Tối đa 10 requests song song

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

        # Chạy tất cả song song (concurrent)
        per_case_results = await asyncio.gather(*[eval_single(c) for c in dataset])

        hit_rates = [r["hit_rate"] for r in per_case_results if r["hit_rate"] is not None]
        mrrs = [r["mrr"] for r in per_case_results if r["mrr"] is not None]
        skipped_adversarial = sum(1 for r in per_case_results if r["hit_rate"] is None)

        # Aggregate metrics
        evaluated = len(hit_rates)
        avg_hit_rate = round(sum(hit_rates) / evaluated, 4) if evaluated else 0.0
        avg_mrr = round(sum(mrrs) / evaluated, 4) if evaluated else 0.0

        # Phân tích theo difficulty
        difficulty_breakdown = {}
        for r in per_case_results:
            if r["hit_rate"] is None:
                continue
            diff = r["difficulty"]
            if diff not in difficulty_breakdown:
                difficulty_breakdown[diff] = {"hit_rates": [], "mrrs": []}
            difficulty_breakdown[diff]["hit_rates"].append(r["hit_rate"])
            difficulty_breakdown[diff]["mrrs"].append(r["mrr"])

        difficulty_summary = {}
        for diff, vals in difficulty_breakdown.items():
            n = len(vals["hit_rates"])
            difficulty_summary[diff] = {
                "count": n,
                "avg_hit_rate": round(sum(vals["hit_rates"]) / n, 4) if n else 0,
                "avg_mrr": round(sum(vals["mrrs"]) / n, 4) if n else 0,
            }

        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "total_evaluated": evaluated,
            "skipped_adversarial": skipped_adversarial,
            "perfect_hit_count": sum(1 for h in hit_rates if h == 1.0),
            "zero_hit_count": sum(1 for h in hit_rates if h == 0.0),
            "difficulty_breakdown": difficulty_summary,
            "per_case": per_case_results,
        }
