import asyncio
import time
from typing import Dict, List

from config import BENCHMARK_MAX_CONCURRENCY


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, max_concurrency: int = BENCHMARK_MAX_CONCURRENCY):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.max_concurrency = max(1, max_concurrency)

    async def run_single_test(self, test_case: Dict) -> Dict:
        started_at = time.perf_counter()

        response = await self.agent.query(test_case["question"])
        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        latency = time.perf_counter() - started_at
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], max_concurrency: int = None) -> List[Dict]:
        """
        Run the full benchmark concurrently with a semaphore guard instead of
        sequential fixed batches. This keeps the pipeline saturated while still
        respecting API rate limits.
        """
        concurrency = max(1, max_concurrency or self.max_concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict] = [None] * len(dataset)

        async def run_index(index: int, case: Dict) -> None:
            async with semaphore:
                results[index] = await self.run_single_test(case)

        await asyncio.gather(*(run_index(index, case) for index, case in enumerate(dataset)))
        return results
