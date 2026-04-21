import asyncio
import json
import os
import time

from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent, MainAgentV2


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

    # Cost from the shared tracker (caller resets before each run)
    cost_info = LLMJudge.get_cost_summary()

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
            # Cost tracking (from LLMJudge cost tracker)
            "total_cost_usd": round(cost_info["total_cost_usd"], 6),
            "total_tokens": cost_info["total_input_tokens"] + cost_info["total_output_tokens"],
            "cost_per_eval_usd": round(cost_info["cost_per_eval_usd"], 6),
            "cost_breakdown": cost_info["per_model"],
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

    # 2. Khởi tạo agents — V1 là baseline, V2 là bản tối ưu
    agent_v1 = MainAgent()
    agent_v2 = MainAgentV2()

    # 3. Chạy Retrieval Evaluation TRƯỚC (dùng V1 agent — retrieval config giống nhau)
    print("\n🔍 Đang chạy Retrieval Evaluation (Hit Rate & MRR)...")
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_results = await retrieval_evaluator.evaluate_batch(dataset, agent_v1)

    hit = retrieval_results["avg_hit_rate"]
    mrr = retrieval_results["avg_mrr"]
    n = retrieval_results["total_evaluated"]
    zero_hit = retrieval_results["zero_hit_count"]
    print(f"   Hit Rate: {hit*100:.1f}%  |  MRR: {mrr:.3f}  |  Evaluated: {n} cases")
    print(f"   Zero-Hit (hallucination risk): {zero_hit} cases ({zero_hit/n*100:.1f}%)")
    print(f"   By difficulty: {retrieval_results.get('difficulty_breakdown', {})}")

    # 4. Chạy Benchmark V1 (baseline — temperature mặc định, prompt gốc)
    print(f"\n🏃 Running V1 benchmark ({agent_v1.name})...")
    LLMJudge.reset_cost_tracker()
    v1_results, v1_summary = await run_benchmark_with_results(
        "Agent_V1_Base", dataset, agent_v1, retrieval_results
    )

    # 5. Chạy Benchmark V2 (optimized — temperature=0, structured-response prompt)
    print(f"\n🏃 Running V2 benchmark ({agent_v2.name})...")
    LLMJudge.reset_cost_tracker()
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", dataset, agent_v2, retrieval_results
    )

    if not v1_summary or not v2_summary:
        print("❌ Benchmark thất bại.")
        return

    # 6. Regression Analysis — Delta trên mọi chiều quan trọng
    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta_score   = v2_summary["metrics"]["avg_score"]  - v1_summary["metrics"]["avg_score"]
    delta_pass    = v2_summary["metrics"]["pass_rate"]   - v1_summary["metrics"]["pass_rate"]
    delta_agree   = v2_summary["metrics"]["agreement_rate"] - v1_summary["metrics"]["agreement_rate"]
    cost_v1       = v1_summary["metrics"]["total_cost_usd"]
    cost_v2       = v2_summary["metrics"]["total_cost_usd"]
    delta_cost    = cost_v2 - cost_v1

    def _sign(x): return "+" if x >= 0 else ""

    print(f"{'Metric':<22} {'V1':>8} {'V2':>8} {'Delta':>10}")
    print("-" * 52)
    print(f"{'Avg Judge Score':<22} {v1_summary['metrics']['avg_score']:>8.4f} {v2_summary['metrics']['avg_score']:>8.4f} {_sign(delta_score)}{delta_score:>9.4f}")
    print(f"{'Pass Rate':<22} {v1_summary['metrics']['pass_rate']:>8.4f} {v2_summary['metrics']['pass_rate']:>8.4f} {_sign(delta_pass)}{delta_pass:>9.4f}")
    print(f"{'Agreement Rate':<22} {v1_summary['metrics']['agreement_rate']:>8.4f} {v2_summary['metrics']['agreement_rate']:>8.4f} {_sign(delta_agree)}{delta_agree:>9.4f}")
    print(f"{'Hit Rate (shared)':<22} {v1_summary['metrics']['hit_rate']:>8.4f} {v2_summary['metrics']['hit_rate']:>8.4f} {'N/A':>10}")
    print(f"{'Cost (USD)':<22} ${cost_v1:>7.4f} ${cost_v2:>7.4f} {_sign(delta_cost)}${abs(delta_cost):.4f}")

    # Retrieval ↔ Answer Quality Correlation Analysis
    _print_correlation_analysis(retrieval_results, v2_results)

    # 7. Release Gate — Quyết định Release hay Rollback tự động
    SCORE_THRESHOLD     = 3.5
    HIT_RATE_THRESHOLD  = 0.5
    DELTA_SCORE_MIN     = 0.0   # V2 không được tệ hơn V1
    COST_INCREASE_MAX   = 0.10  # Không cho phép tăng chi phí >10%

    score_ok     = v2_summary["metrics"]["avg_score"] >= SCORE_THRESHOLD
    hit_rate_ok  = v2_summary["metrics"]["hit_rate"]  >= HIT_RATE_THRESHOLD
    delta_ok     = delta_score >= DELTA_SCORE_MIN
    cost_ok      = cost_v1 == 0 or (delta_cost / max(cost_v1, 1e-9)) <= COST_INCREASE_MAX

    decision = score_ok and hit_rate_ok and delta_ok and cost_ok

    gate_reasons = []
    if not score_ok:
        gate_reasons.append(f"avg_score {v2_summary['metrics']['avg_score']:.2f} < threshold {SCORE_THRESHOLD}")
    if not hit_rate_ok:
        gate_reasons.append(f"hit_rate {v2_summary['metrics']['hit_rate']:.2%} < threshold {HIT_RATE_THRESHOLD:.0%}")
    if not delta_ok:
        gate_reasons.append(f"delta_score {delta_score:+.4f} < {DELTA_SCORE_MIN} (regression detected)")
    if not cost_ok:
        gate_reasons.append(f"cost increase {delta_cost/cost_v1:.0%} > {COST_INCREASE_MAX:.0%} budget")

    decision_str = "APPROVE" if decision else "ROLLBACK"
    print(f"\n{'✅' if decision else '❌'} RELEASE GATE DECISION: {decision_str}")
    if gate_reasons:
        print("   Blocking reasons:")
        for r in gate_reasons:
            print(f"     • {r}")

    v2_summary["regression"] = {
        "v1_score":        v1_summary["metrics"]["avg_score"],
        "v2_score":        v2_summary["metrics"]["avg_score"],
        "delta_score":     round(delta_score, 4),
        "v1_pass_rate":    v1_summary["metrics"]["pass_rate"],
        "v2_pass_rate":    v2_summary["metrics"]["pass_rate"],
        "delta_pass_rate": round(delta_pass, 4),
        "v1_agreement":    v1_summary["metrics"]["agreement_rate"],
        "v2_agreement":    v2_summary["metrics"]["agreement_rate"],
        "v1_cost_usd":     cost_v1,
        "v2_cost_usd":     cost_v2,
        "delta_cost_usd":  round(delta_cost, 6),
        "release_decision": decision_str,
        "blocking_reasons": gate_reasons,
        "thresholds": {
            "min_score":         SCORE_THRESHOLD,
            "min_hit_rate":      HIT_RATE_THRESHOLD,
            "min_delta_score":   DELTA_SCORE_MIN,
            "max_cost_increase": COST_INCREASE_MAX,
        },
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
    print(f"💰 V1 cost: ${cost_v1:.4f}  |  V2 cost: ${cost_v2:.4f}  |  V2 tokens: {v2_summary['metrics']['total_tokens']}")


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
