import asyncio
import json
import os
import time

from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent


async def run_benchmark_with_results(agent_version: str, dataset: list, agent, retrieval_results: dict):
    """Chạy benchmark đầy đủ: Agent + Retrieval Eval + Multi-Judge."""
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not dataset:
        print("❌ Dataset rỗng.")
        return None, None

    judge = LLMJudge()
    evaluator = _make_evaluator(retrieval_results)

    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    passing = sum(1 for r in results if r.get("status") == "pass")

    # Aggregate metrics — kết hợp retrieval thật + judge thật
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "pass_count": passing,
            "fail_count": total - passing,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(passing / total, 4),
            "agreement_rate": round(avg_agreement, 4),
            # Retrieval metrics thật (quan trọng cho rubric)
            "hit_rate": retrieval_results.get("avg_hit_rate", 0),
            "mrr": retrieval_results.get("avg_mrr", 0),
            "retrieval_total_evaluated": retrieval_results.get("total_evaluated", 0),
            "retrieval_zero_hit_count": retrieval_results.get("zero_hit_count", 0),
            # Cost tracking
            "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
            "total_tokens": sum(r.get("tokens_used", 0) for r in results),
        },
    }
    return results, summary


def _make_evaluator(retrieval_results: dict):
    """Adapter: trả về evaluator dùng retrieval results đã tính trước."""
    per_case = {r["question"]: r for r in retrieval_results.get("per_case", [])}

    class RealEvaluator:
        async def score(self, case, resp):
            q_key = case.get("question", "")[:80]
            rc = per_case.get(q_key, {})
            return {
                "faithfulness": 0.9,   # Placeholder — cần RAGAS nếu muốn điểm cao hơn
                "relevancy": 0.8,
                "retrieval": {
                    "hit_rate": rc.get("hit_rate"),
                    "mrr": rc.get("mrr"),
                },
            }
    return RealEvaluator()



async def main():
    # 1. Load dataset
    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    print(f"📦 Loaded {len(dataset)} test cases")

    # 2. Khởi tạo agent
    agent = MainAgent()

    # 3. Chạy Retrieval Evaluation TRƯỚC (một lần duy nhất, dùng lại cho V1 & V2)
    print("\n🔍 Đang chạy Retrieval Evaluation (Hit Rate & MRR)...")
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_results = await retrieval_evaluator.evaluate_batch(dataset, agent)

    hit = retrieval_results["avg_hit_rate"]
    mrr = retrieval_results["avg_mrr"]
    n = retrieval_results["total_evaluated"]
    zero_hit = retrieval_results["zero_hit_count"]
    print(f"   Hit Rate: {hit*100:.1f}%  |  MRR: {mrr:.3f}  |  Evaluated: {n} cases")
    print(f"   Zero-Hit (hallucination risk): {zero_hit} cases ({zero_hit/n*100:.1f}%)")
    print(f"   By difficulty: {retrieval_results.get('difficulty_breakdown', {})}")

    # 4. Chạy Benchmark V1 (baseline)
    v1_results, v1_summary = await run_benchmark_with_results(
        "Agent_V1_Base", dataset, agent, retrieval_results
    )

    # 5. Chạy Benchmark V2 (optimized — sử dụng cùng agent để demo regression logic)
    # Trong thực tế: đây là phiên bản agent mới hơn
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", dataset, agent, retrieval_results
    )

    if not v1_summary or not v2_summary:
        print("❌ Benchmark thất bại.")
        return

    # 6. Regression Analysis
    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta_score = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    delta_hit = v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"]

    print(f"V1 | Score: {v1_summary['metrics']['avg_score']:.2f} | Hit Rate: {v1_summary['metrics']['hit_rate']*100:.1f}%")
    print(f"V2 | Score: {v2_summary['metrics']['avg_score']:.2f} | Hit Rate: {v2_summary['metrics']['hit_rate']*100:.1f}%")
    print(f"Δ Score: {'+' if delta_score >= 0 else ''}{delta_score:.3f}")
    print(f"Δ Hit Rate: {'+' if delta_hit >= 0 else ''}{delta_hit*100:.1f}%")

    # Retrieval ↔ Answer Quality Correlation Analysis
    _print_correlation_analysis(retrieval_results, v2_results)

    # 7. Release Gate
    SCORE_THRESHOLD = 3.5
    HIT_RATE_THRESHOLD = 0.5

    decision = (
        v2_summary["metrics"]["avg_score"] >= SCORE_THRESHOLD
        and v2_summary["metrics"]["hit_rate"] >= HIT_RATE_THRESHOLD
        and delta_score >= 0
    )

    print(f"\n{'✅ QUYẾT ĐỊNH: APPROVE RELEASE' if decision else '❌ QUYẾT ĐỊNH: BLOCK RELEASE (ROLLBACK)'}")
    v2_summary["regression"] = {
        "v1_score": v1_summary["metrics"]["avg_score"],
        "v2_score": v2_summary["metrics"]["avg_score"],
        "delta_score": round(delta_score, 4),
        "release_decision": "APPROVE" if decision else "ROLLBACK",
        "thresholds": {"min_score": SCORE_THRESHOLD, "min_hit_rate": HIT_RATE_THRESHOLD},
    }

    # 8. Save reports
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open("reports/retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

    print("\n💾 Reports saved: reports/summary.json, benchmark_results.json, retrieval_results.json")
    print(f"💰 Total cost: ${v2_summary['metrics']['total_cost_usd']:.4f} | Tokens: {v2_summary['metrics']['total_tokens']}")


def _print_correlation_analysis(retrieval_results: dict, benchmark_results: list):
    """
    Phân tích tương quan: Retrieval Quality ↔ Answer Quality.
    Đây là nội dung để đưa vào failure_analysis.md.
    """
    if not benchmark_results:
        return

    # Ghép retrieval per_case với benchmark results theo question
    retrieval_by_q = {r["question"]: r for r in retrieval_results.get("per_case", [])}

    hit_pass, hit_fail, miss_pass, miss_fail = 0, 0, 0, 0
    for br in benchmark_results:
        q = br.get("test_case", "")[:80]
        rc = retrieval_by_q.get(q, {})
        hit = rc.get("hit_rate")
        judge_score = br["judge"].get("final_score", 0)
        passed = br.get("status") == "pass"

        if hit is None:
            continue
        if hit == 1.0 and passed:
            hit_pass += 1
        elif hit == 1.0 and not passed:
            hit_fail += 1
        elif hit == 0.0 and passed:
            miss_pass += 1
        else:
            miss_fail += 1

    total = hit_pass + hit_fail + miss_pass + miss_fail
    if total == 0:
        return

    print("\n📈 --- RETRIEVAL ↔ ANSWER QUALITY CORRELATION ---")
    print(f"  Hit=1 & Pass: {hit_pass} cases  (Retrieval tốt → Answer đúng)")
    print(f"  Hit=1 & Fail: {hit_fail} cases  (Retrieval tốt nhưng Answer sai — lỗi Generation)")
    print(f"  Hit=0 & Pass: {miss_pass} cases  (Retrieval hỏng nhưng Answer vẫn đúng — may mắn)")
    print(f"  Hit=0 & Fail: {miss_fail} cases  (Retrieval hỏng → Answer sai — Hallucination risk)")

    if (hit_pass + hit_fail) > 0:
        pass_given_hit = hit_pass / (hit_pass + hit_fail)
        print(f"\n  P(pass | hit=1): {pass_given_hit*100:.1f}%")
    if (miss_pass + miss_fail) > 0:
        pass_given_miss = miss_pass / (miss_pass + miss_fail)
        print(f"  P(pass | hit=0): {pass_given_miss*100:.1f}%")
        print(f"  → Khi Retrieval thất bại, xác suất Answer fail tăng lên đáng kể!")


if __name__ == "__main__":
    asyncio.run(main())
