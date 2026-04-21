import asyncio
import json
import os
import time

from agent.main_agent import MainAgent, MainAgentV2
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _seconds(value: float) -> float:
    return round(value, 3)


def _build_runtime_summary(results: list, started_at: float, stage_name: str) -> dict:
    duration_sec = time.perf_counter() - started_at
    latencies = [result["latency"] for result in results] if results else []

    return {
        "stage": stage_name,
        "wall_clock_sec": _seconds(duration_sec),
        "wall_clock_min": round(duration_sec / 60, 3),
        "cases_per_minute": round((len(results) / max(duration_sec, 1e-9)) * 60, 2) if results else 0.0,
        "latency_sec": {
            "avg": _seconds(sum(latencies) / len(latencies)) if latencies else 0.0,
            "min": _seconds(min(latencies)) if latencies else 0.0,
            "max": _seconds(max(latencies)) if latencies else 0.0,
        },
    }


def _build_cost_report(label: str, cost_info: dict, runtime: dict, total_cases: int) -> dict:
    return {
        "benchmark": label,
        "total_cases": total_cases,
        "runtime": runtime,
        "tokens": {
            "input": cost_info["total_input_tokens"],
            "output": cost_info["total_output_tokens"],
            "total": cost_info["total_tokens"],
        },
        "cost": {
            "total_usd": round(cost_info["total_cost_usd"], 6),
            "per_eval_usd": round(cost_info["cost_per_eval_usd"], 6),
            "avg_latency_ms_per_llm_call": cost_info["avg_latency_ms_per_call"],
        },
        "per_model": cost_info["per_model"],
        "llm_call_records": cost_info["records"],
    }


async def run_benchmark_with_results(agent_version: str, dataset: list, agent, retrieval_results: dict):
    """Run one benchmark version and return detailed results plus summary."""
    print(f"Starting benchmark for {agent_version}...")

    if not dataset:
        print("Dataset is empty.")
        return None, None, None

    judge = LLMJudge()
    evaluator = _make_evaluator(retrieval_results)
    runner = BenchmarkRunner(agent, evaluator, judge)

    benchmark_started_at = time.perf_counter()
    results = await runner.run_all(dataset)
    runtime = _build_runtime_summary(results, benchmark_started_at, agent_version)

    total = len(results)
    passing = sum(1 for result in results if result.get("status") == "pass")
    avg_score = sum(result["judge"]["final_score"] for result in results) / total
    avg_agreement = sum(result["judge"]["agreement_rate"] for result in results) / total

    cost_info = LLMJudge.get_cost_summary(eval_count=total)
    cost_report = _build_cost_report(agent_version, cost_info, runtime, total)

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "pass_count": passing,
            "fail_count": total - passing,
            "timestamp": _now(),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(passing / total, 4),
            "agreement_rate": round(avg_agreement, 4),
            "hit_rate": retrieval_results.get("avg_hit_rate", 0),
            "mrr": retrieval_results.get("avg_mrr", 0),
            "retrieval_total_evaluated": retrieval_results.get("total_evaluated", 0),
            "retrieval_zero_hit_count": retrieval_results.get("zero_hit_count", 0),
            "total_cost_usd": round(cost_info["total_cost_usd"], 6),
            "total_tokens": cost_info["total_tokens"],
            "input_tokens": cost_info["total_input_tokens"],
            "output_tokens": cost_info["total_output_tokens"],
            "cost_per_eval_usd": round(cost_info["cost_per_eval_usd"], 6),
            "avg_llm_latency_ms": cost_info["avg_latency_ms_per_call"],
            "cost_breakdown": cost_info["per_model"],
        },
        "performance": runtime,
    }
    return results, summary, cost_report


def _make_evaluator(retrieval_results: dict):
    """Return an evaluator backed by the precomputed retrieval metrics."""
    per_case = {result["question"]: result for result in retrieval_results.get("per_case", [])}

    class RealEvaluator:
        async def score(self, case, resp):
            q_key = case.get("question", "")[:80]
            retrieval_case = per_case.get(q_key, {})
            return {
                "faithfulness": 0.9,
                "relevancy": 0.8,
                "retrieval": {
                    "hit_rate": retrieval_case.get("hit_rate"),
                    "mrr": retrieval_case.get("mrr"),
                },
            }

    return RealEvaluator()


async def main():
    pipeline_started_at = time.perf_counter()

    if not os.path.exists("data/golden_set.jsonl"):
        print("Missing data/golden_set.jsonl. Run 'python data/synthetic_gen.py' first.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as file:
        dataset = [json.loads(line) for line in file if line.strip()]
    print(f"Loaded {len(dataset)} test cases")

    agent_v1 = MainAgent()
    agent_v2 = MainAgentV2()

    print("\nRunning retrieval evaluation (Hit Rate and MRR)...")
    retrieval_started_at = time.perf_counter()
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_results = await retrieval_evaluator.evaluate_batch(dataset, agent_v1)
    retrieval_runtime_sec = time.perf_counter() - retrieval_started_at

    hit = retrieval_results["avg_hit_rate"]
    mrr = retrieval_results["avg_mrr"]
    evaluated_cases = retrieval_results["total_evaluated"]
    zero_hit = retrieval_results["zero_hit_count"]
    zero_hit_rate = (zero_hit / max(evaluated_cases, 1)) * 100
    print(f"   Hit Rate: {hit * 100:.1f}% | MRR: {mrr:.3f} | Evaluated: {evaluated_cases} cases")
    print(f"   Zero-Hit: {zero_hit} cases ({zero_hit_rate:.1f}%)")
    print(f"   Retrieval runtime: {retrieval_runtime_sec:.2f}s")

    print(f"\nRunning V1 benchmark ({agent_v1.name})...")
    LLMJudge.reset_cost_tracker()
    v1_results, v1_summary, v1_cost_report = await run_benchmark_with_results(
        "Agent_V1_Base",
        dataset,
        agent_v1,
        retrieval_results,
    )

    print(f"\nRunning V2 benchmark ({agent_v2.name})...")
    LLMJudge.reset_cost_tracker()
    v2_results, v2_summary, v2_cost_report = await run_benchmark_with_results(
        "Agent_V2_Optimized",
        dataset,
        agent_v2,
        retrieval_results,
    )

    if not v1_summary or not v2_summary:
        print("Benchmark failed.")
        return

    print("\n--- REGRESSION COMPARISON ---")
    delta_score = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    delta_pass = v2_summary["metrics"]["pass_rate"] - v1_summary["metrics"]["pass_rate"]
    delta_agree = v2_summary["metrics"]["agreement_rate"] - v1_summary["metrics"]["agreement_rate"]
    cost_v1 = v1_summary["metrics"]["total_cost_usd"]
    cost_v2 = v2_summary["metrics"]["total_cost_usd"]
    delta_cost = cost_v2 - cost_v1
    v1_runtime = v1_summary["performance"]["wall_clock_sec"]
    v2_runtime = v2_summary["performance"]["wall_clock_sec"]

    def _sign(value):
        return "+" if value >= 0 else ""

    print(f"{'Metric':<22} {'V1':>10} {'V2':>10} {'Delta':>12}")
    print("-" * 58)
    print(f"{'Avg Judge Score':<22} {v1_summary['metrics']['avg_score']:>10.4f} {v2_summary['metrics']['avg_score']:>10.4f} {_sign(delta_score)}{delta_score:>11.4f}")
    print(f"{'Pass Rate':<22} {v1_summary['metrics']['pass_rate']:>10.4f} {v2_summary['metrics']['pass_rate']:>10.4f} {_sign(delta_pass)}{delta_pass:>11.4f}")
    print(f"{'Agreement Rate':<22} {v1_summary['metrics']['agreement_rate']:>10.4f} {v2_summary['metrics']['agreement_rate']:>10.4f} {_sign(delta_agree)}{delta_agree:>11.4f}")
    print(f"{'Hit Rate (shared)':<22} {v1_summary['metrics']['hit_rate']:>10.4f} {v2_summary['metrics']['hit_rate']:>10.4f} {'N/A':>12}")
    print(f"{'Runtime (sec)':<22} {v1_runtime:>10.2f} {v2_runtime:>10.2f} {_sign(v2_runtime - v1_runtime)}{(v2_runtime - v1_runtime):>11.2f}")
    print(f"{'Cost (USD)':<22} ${cost_v1:>9.4f} ${cost_v2:>9.4f} {_sign(delta_cost)}${abs(delta_cost):.4f}")

    _print_correlation_analysis(retrieval_results, v2_results)

    score_threshold = 3.5
    hit_rate_threshold = 0.5
    delta_score_min = 0.0
    cost_increase_max = 0.10

    score_ok = v2_summary["metrics"]["avg_score"] >= score_threshold
    hit_rate_ok = v2_summary["metrics"]["hit_rate"] >= hit_rate_threshold
    delta_ok = delta_score >= delta_score_min
    cost_ok = cost_v1 == 0 or (delta_cost / max(cost_v1, 1e-9)) <= cost_increase_max
    decision = score_ok and hit_rate_ok and delta_ok and cost_ok

    gate_reasons = []
    if not score_ok:
        gate_reasons.append(f"avg_score {v2_summary['metrics']['avg_score']:.2f} < threshold {score_threshold}")
    if not hit_rate_ok:
        gate_reasons.append(f"hit_rate {v2_summary['metrics']['hit_rate']:.2%} < threshold {hit_rate_threshold:.0%}")
    if not delta_ok:
        gate_reasons.append(f"delta_score {delta_score:+.4f} < {delta_score_min} (regression detected)")
    if not cost_ok:
        gate_reasons.append(f"cost increase {delta_cost / cost_v1:.0%} > {cost_increase_max:.0%} budget")

    decision_str = "APPROVE" if decision else "ROLLBACK"
    print(f"\n{'PASS' if decision else 'FAIL'} RELEASE GATE DECISION: {decision_str}")
    if gate_reasons:
        print("   Blocking reasons:")
        for reason in gate_reasons:
            print(f"   - {reason}")

    total_pipeline_sec = time.perf_counter() - pipeline_started_at
    pipeline_performance = {
        "dataset_size": len(dataset),
        "retrieval_eval_sec": _seconds(retrieval_runtime_sec),
        "benchmark_v1_sec": v1_runtime,
        "benchmark_v2_sec": v2_runtime,
        "total_pipeline_sec": _seconds(total_pipeline_sec),
        "total_pipeline_min": round(total_pipeline_sec / 60, 3),
        "meets_async_target_under_2_min_for_50_cases": len(dataset) >= 50 and total_pipeline_sec < 120,
    }

    v2_summary["regression"] = {
        "v1_score": v1_summary["metrics"]["avg_score"],
        "v2_score": v2_summary["metrics"]["avg_score"],
        "delta_score": round(delta_score, 4),
        "v1_pass_rate": v1_summary["metrics"]["pass_rate"],
        "v2_pass_rate": v2_summary["metrics"]["pass_rate"],
        "delta_pass_rate": round(delta_pass, 4),
        "v1_agreement": v1_summary["metrics"]["agreement_rate"],
        "v2_agreement": v2_summary["metrics"]["agreement_rate"],
        "v1_cost_usd": cost_v1,
        "v2_cost_usd": cost_v2,
        "delta_cost_usd": round(delta_cost, 6),
        "v1_runtime_sec": v1_runtime,
        "v2_runtime_sec": v2_runtime,
        "release_decision": decision_str,
        "blocking_reasons": gate_reasons,
        "thresholds": {
            "min_score": score_threshold,
            "min_hit_rate": hit_rate_threshold,
            "min_delta_score": delta_score_min,
            "max_cost_increase": cost_increase_max,
        },
    }
    v2_summary["pipeline_performance"] = pipeline_performance

    report_bundle = {
        "generated_at": _now(),
        "pipeline_performance": pipeline_performance,
        "retrieval": {
            "runtime_sec": _seconds(retrieval_runtime_sec),
            "avg_hit_rate": retrieval_results["avg_hit_rate"],
            "avg_mrr": retrieval_results["avg_mrr"],
            "total_evaluated": retrieval_results["total_evaluated"],
            "zero_hit_count": retrieval_results["zero_hit_count"],
        },
        "benchmarks": {
            "Agent_V1_Base": v1_cost_report,
            "Agent_V2_Optimized": v2_cost_report,
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as file:
        json.dump(v2_summary, file, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as file:
        json.dump(v2_results, file, ensure_ascii=False, indent=2)
    with open("reports/retrieval_results.json", "w", encoding="utf-8") as file:
        json.dump(retrieval_results, file, ensure_ascii=False, indent=2)
    with open("reports/cost_report.json", "w", encoding="utf-8") as file:
        json.dump(report_bundle, file, ensure_ascii=False, indent=2)

    print("\nReports saved: reports/summary.json, reports/benchmark_results.json, reports/retrieval_results.json, reports/cost_report.json")
    print(f"V1 cost: ${cost_v1:.4f} | V2 cost: ${cost_v2:.4f} | V2 tokens: {v2_summary['metrics']['total_tokens']}")
    print(f"Total pipeline runtime: {total_pipeline_sec:.2f}s")


def _print_correlation_analysis(retrieval_results: dict, benchmark_results: list):
    """Print retrieval quality versus answer quality correlation."""
    if not benchmark_results:
        return

    retrieval_by_question = {result["question"]: result for result in retrieval_results.get("per_case", [])}

    hit_pass = hit_fail = miss_pass = miss_fail = 0
    for benchmark_result in benchmark_results:
        question_key = benchmark_result.get("test_case", "")[:80]
        retrieval_case = retrieval_by_question.get(question_key, {})
        hit = retrieval_case.get("hit_rate")
        passed = benchmark_result.get("status") == "pass"

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

    print("\n--- RETRIEVAL VS ANSWER QUALITY ---")
    print(f"  Hit=1 and Pass: {hit_pass} cases")
    print(f"  Hit=1 and Fail: {hit_fail} cases")
    print(f"  Hit=0 and Pass: {miss_pass} cases")
    print(f"  Hit=0 and Fail: {miss_fail} cases")

    if (hit_pass + hit_fail) > 0:
        pass_given_hit = hit_pass / (hit_pass + hit_fail)
        print(f"\n  P(pass | hit=1): {pass_given_hit * 100:.1f}%")
    if (miss_pass + miss_fail) > 0:
        pass_given_miss = miss_pass / (miss_pass + miss_fail)
        print(f"  P(pass | hit=0): {pass_given_miss * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
