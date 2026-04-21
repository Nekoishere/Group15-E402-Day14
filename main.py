import asyncio
import json
import os
import time

from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent


async def run_benchmark_with_results(agent_version: str):
    print(f"\n🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    print(f"📂 Đã load {len(dataset)} test cases.")

    # ── Initialize REAL components ────────────────────────────────────────────
    agent = MainAgent()
    evaluator = RetrievalEvaluator(top_k=5)
    judge = LLMJudge()

    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    pass_count = sum(1 for r in results if r["status"] == "pass")

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "passed": pass_count,
            "failed": total - pass_count,
            "pass_rate": round(pass_count / total, 4) if total else 0,
            "judges": ["gpt-4o-mini", "gemini-2.5-flash", "meta-judge:gpt-4o-mini"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(
                sum(r["judge"]["final_score"] for r in results) / total, 4
            ),
            "hit_rate": round(
                sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total, 4
            ),
            "mrr": round(
                sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total, 4
            ),
            "avg_faithfulness": round(
                sum(r["ragas"]["faithfulness"] for r in results) / total, 4
            ),
            "avg_relevancy": round(
                sum(r["ragas"]["relevancy"] for r in results) / total, 4
            ),
            "agreement_rate": round(
                sum(r["judge"]["agreement_rate"] for r in results) / total, 4
            ),
            "cohens_kappa": judge.compute_batch_kappa(),
            "avg_latency_sec": round(
                sum(r["latency"] for r in results) / total, 3
            ),
        },
    }
    return results, summary


async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary


async def main():
    # ── V1: Base agent run ────────────────────────────────────────────────────
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")

    if not v1_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    # ── V2: Simulated optimized version (same agent, for regression gate demo) ─
    # In a real scenario this would point to a different agent configuration.
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v2_summary:
        return

    # ── Regression Analysis ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📊  KẾT QUẢ SO SÁNH (REGRESSION GATE)")
    print("=" * 55)

    v1m = v1_summary["metrics"]
    v2m = v2_summary["metrics"]

    delta_score = v2m["avg_score"] - v1m["avg_score"]
    delta_hit = v2m["hit_rate"] - v1m["hit_rate"]
    delta_agreement = v2m["agreement_rate"] - v1m["agreement_rate"]

    def fmt(val, pct=False):
        s = f"{val*100:.1f}%" if pct else f"{val:.4f}"
        return ("+" if val >= 0 else "") + s

    print(f"{'Metric':<22} {'V1':>10} {'V2':>10} {'Delta':>10}")
    print("-" * 55)
    print(f"{'Avg Judge Score':<22} {v1m['avg_score']:>10.4f} {v2m['avg_score']:>10.4f} {fmt(delta_score):>10}")
    print(f"{'Hit Rate @5':<22} {v1m['hit_rate']:>10.1%} {v2m['hit_rate']:>10.1%} {fmt(delta_hit, pct=True):>10}")
    print(f"{'MRR':<22} {v1m['mrr']:>10.4f} {v2m['mrr']:>10.4f} {fmt(v2m['mrr']-v1m['mrr']):>10}")
    print(f"{'Agreement Rate':<22} {v1m['agreement_rate']:>10.1%} {v2m['agreement_rate']:>10.1%} {fmt(delta_agreement, pct=True):>10}")
    print(f"{'Cohen\\'s Kappa':<22} {v1m['cohens_kappa']:>10.4f} {v2m['cohens_kappa']:>10.4f} {fmt(v2m['cohens_kappa']-v1m['cohens_kappa']):>10}")
    print(f"{'Avg Latency (s)':<22} {v1m['avg_latency_sec']:>10.3f} {v2m['avg_latency_sec']:>10.3f}")
    print("=" * 55)

    # ── Auto-Gate decision ────────────────────────────────────────────────────
    if delta_score >= 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE RELEASE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI PHÁT HÀNH (BLOCK RELEASE) — Điểm bị giảm!")

    # ── Report Cost and Tokens Summary ───────────────────────────────────────
    cost_summary = LLMJudge.get_cost_summary()
    print("\n💰  CHI PHÍ VÀ TOKEN USAGE (Multi-Judge)")
    print(f"Tổng số API calls: {cost_summary['total_calls']}")
    print(f"Tổng Input Tokens: {cost_summary['total_input_tokens']}")
    print(f"Tổng Output Tokens: {cost_summary['total_output_tokens']}")
    print(f"Tổng Chi phí (Toàn bộ Benchmark): ${cost_summary['total_cost_usd']:.6f}")
    print(f"Chi phí trung bình / 1 Test case: ${cost_summary['cost_per_eval_usd']:.6f}")
    
    # Update v2 summary with cost
    v2_summary["cost_and_tokens"] = cost_summary

    # ── Save reports ──────────────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Reports saved to reports/summary.json & reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
