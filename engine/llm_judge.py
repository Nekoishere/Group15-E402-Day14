"""
Multi-Judge Consensus Engine – Day 14 AI Evaluation
=====================================================
Architecture:
  - Judge A : GPT-4o-mini  (OpenAI)
  - Judge B : Gemini 2.0 Flash (Google)

Features implemented:
  1. Concurrent dual-judge scoring (async)
  2. Agreement Rate (percentage of exact matches over a batch)
  3. Cohen's Kappa (inter-rater reliability, corrects for chance agreement)
  4. Conflict Resolution: tie-breaker via a third "meta-judge" call (GPT-4o-mini)
     when score difference > threshold
  5. Position Bias Check: swap answer order to detect positional preference
  6. Cost & Token Usage tracking per call and aggregated over a run
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─── Client initialization ────────────────────────────────────────────────────
_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

# ─── Pricing (USD per 1K tokens, approximate) ────────────────────────────────
COST_TABLE = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},   # per 1K tokens
    "gemini-2.0-flash": {"input": 0.000100, "output": 0.000400},
}

# ─── Judge Prompt ─────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are an expert AI evaluator for VinUniversity's academic chatbot (VinLex).
Your task is to objectively score the AI response against the ground truth answer.

Score on a scale of 1 to 5:
  1 = Completely wrong / hallucinated / irrelevant
  2 = Mostly incorrect or critically incomplete
  3 = Partially correct — key information present but has noticeable gaps or errors
  4 = Mostly correct with only minor issues
  5 = Fully accurate, complete, and professionally worded

Evaluation criteria:
  - Accuracy   : Is the answer factually correct based on the ground truth?
  - Completeness: Does it cover the key points from the ground truth?
  - Tone       : Is the language professional and appropriate for a student-facing chatbot?

---
Question: {question}
Ground Truth: {ground_truth}
AI Response: {answer}
---

Respond ONLY with valid JSON, no extra text:
{{"score": <integer 1-5>, "reason": "<one concise sentence justification>"}}"""

TIE_BREAKER_PROMPT = """\
Two AI judges produced conflicting scores for a chatbot evaluation:
  - Judge A (GPT-4o-mini) gave score {score_a} with reason: "{reason_a}"
  - Judge B (Gemini-2.0-Flash) gave score {score_b} with reason: "{reason_b}"

Difference = {diff} points. You are the Meta-Judge. Re-read the original evaluation:
Question: {question}
Ground Truth: {ground_truth}
AI Response: {answer}

Decide the FINAL score (integer 1-5) and state WHICH judge's reasoning is more accurate.

Respond ONLY with valid JSON:
{{"final_score": <integer 1-5>, "ruling": "<one sentence on which judge was more accurate and why>"}}"""


# ─── Token cost tracker ───────────────────────────────────────────────────────
class CostTracker:
    """Accumulates token usage and calculates USD cost."""

    def __init__(self):
        self._records: List[Dict] = []

    def add(self, model: str, input_tokens: int, output_tokens: int, latency_ms: float):
        rate = COST_TABLE.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1000
        self._records.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": round(latency_ms, 1),
        })

    def summary(self, eval_count: int = None) -> Dict:
        total_cost = sum(r["cost_usd"] for r in self._records)
        total_input = sum(r["input_tokens"] for r in self._records)
        total_output = sum(r["output_tokens"] for r in self._records)
        total_latency_ms = sum(r["latency_ms"] for r in self._records)
        per_model: Dict[str, Dict] = {}
        for r in self._records:
            m = r["model"]
            if m not in per_model:
                per_model[m] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms_total": 0.0,
                }
            per_model[m]["calls"] += 1
            per_model[m]["input_tokens"] += r["input_tokens"]
            per_model[m]["output_tokens"] += r["output_tokens"]
            per_model[m]["cost_usd"] += r["cost_usd"]
            per_model[m]["latency_ms_total"] += r["latency_ms"]

        for stats in per_model.values():
            calls = max(stats["calls"], 1)
            stats["cost_usd"] = round(stats["cost_usd"], 6)
            stats["avg_input_tokens"] = round(stats["input_tokens"] / calls, 2)
            stats["avg_output_tokens"] = round(stats["output_tokens"] / calls, 2)
            stats["avg_latency_ms"] = round(stats["latency_ms_total"] / calls, 1)
            stats["latency_ms_total"] = round(stats["latency_ms_total"], 1)

        effective_eval_count = eval_count if eval_count is not None else len(self._records)
        return {
            "total_calls": len(self._records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "cost_per_eval_usd": round(total_cost / max(effective_eval_count, 1), 6),
            "avg_latency_ms_per_call": round(total_latency_ms / max(len(self._records), 1), 1),
            "per_model": per_model,
            "records": list(self._records),
        }


# Shared tracker across all LLMJudge instances in a run
_cost_tracker = CostTracker()


# ─── Judge call helpers ───────────────────────────────────────────────────────

async def _call_openai_judge(
    question: str, answer: str, ground_truth: str, prompt_template: str = JUDGE_PROMPT
) -> Tuple[Dict[str, Any], int, int]:
    """Call GPT-4o-mini judge. Returns (result_dict, input_tokens, output_tokens)."""
    prompt = prompt_template.format(
        question=question, ground_truth=ground_truth, answer=answer
    )
    t0 = time.perf_counter()
    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        _cost_tracker.add("gpt-4o-mini", in_tok, out_tok, latency_ms)

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return (
            {
                "model": "gpt-4o-mini",
                "score": max(1, min(5, int(parsed.get("score", 3)))),
                "reason": parsed.get("reason", parsed.get("ruling", "")),
                "latency_ms": round(latency_ms, 1),
            },
            in_tok,
            out_tok,
        )
    except Exception as e:
        return {"model": "gpt-4o-mini", "score": 3, "reason": f"Error: {e}", "latency_ms": 0}, 0, 0


async def _call_gemini_judge(
    question: str, answer: str, ground_truth: str
) -> Tuple[Dict[str, Any], int, int]:
    """Call Gemini 2.5 Flash judge. Returns (result_dict, input_tokens, output_tokens)."""
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    )
    t0 = time.perf_counter()
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=200,
                response_mime_type="application/json",
            ),
        )
        response = await asyncio.to_thread(model.generate_content, prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Token counts from usage_metadata
        meta = getattr(response, "usage_metadata", None)
        in_tok = getattr(meta, "prompt_token_count", 0) or 0
        out_tok = getattr(meta, "candidates_token_count", 0) or 0
        _cost_tracker.add("gemini-2.0-flash", in_tok, out_tok, latency_ms)

        raw = response.text or "{}"
        parsed = json.loads(raw)
        return (
            {
                "model": "gemini-2.0-flash",
                "score": max(1, min(5, int(parsed.get("score", 3)))),
                "reason": parsed.get("reason", ""),
                "latency_ms": round(latency_ms, 1),
            },
            in_tok,
            out_tok,
        )
    except Exception as e:
        return {"model": "gemini-2.0-flash", "score": 3, "reason": f"Error: {e}", "latency_ms": 0}, 0, 0


async def _call_tiebreaker(
    question: str, answer: str, ground_truth: str,
    score_a: int, reason_a: str,
    score_b: int, reason_b: str,
) -> Dict[str, Any]:
    """Meta-judge (GPT-4o-mini) to resolve conflicts when |score_a - score_b| > 1.

    Calls the API directly instead of routing through _call_openai_judge.
    _call_openai_judge applies a second .format() pass on its prompt_template,
    which raises KeyError when judge reasons contain literal { } characters.
    """
    prompt = TIE_BREAKER_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer,
        score_a=score_a, reason_a=reason_a,
        score_b=score_b, reason_b=reason_b,
        diff=abs(score_a - score_b),
    )
    t0 = time.perf_counter()
    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        _cost_tracker.add("gpt-4o-mini", in_tok, out_tok, latency_ms)

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return {
            "final_score": max(1, min(5, int(parsed.get("final_score", (score_a + score_b) // 2)))),
            "ruling": parsed.get("ruling", ""),
        }
    except Exception as e:
        return {"final_score": (score_a + score_b) // 2, "ruling": f"Tie-breaker error: {e}"}


# ─── Cohen's Kappa ────────────────────────────────────────────────────────────

def compute_cohens_kappa(scores_a: List[int], scores_b: List[int], n_categories: int = 5) -> float:
    """
    Compute Cohen's Kappa for inter-rater reliability.
    κ = (P_o - P_e) / (1 - P_e)
    where P_o = observed agreement, P_e = expected agreement by chance.
    Range: -1 to 1. κ > 0.6 = substantial agreement.
    """
    if len(scores_a) != len(scores_b) or not scores_a:
        return 0.0

    n = len(scores_a)
    # Observed agreement
    p_o = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n

    # Expected agreement by chance
    from collections import Counter
    count_a = Counter(scores_a)
    count_b = Counter(scores_b)
    categories = range(1, n_categories + 1)
    p_e = sum((count_a.get(c, 0) / n) * (count_b.get(c, 0) / n) for c in categories)

    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


# ─── Position Bias Check ──────────────────────────────────────────────────────

async def check_position_bias(
    question: str, answer_good: str, answer_bad: str, ground_truth: str
) -> Dict[str, Any]:
    """
    Swap the two answers and check if the judge scores them consistently.
    Calls judges with (A=good, B=bad) then (A=bad, B=good).
    If scores flip significantly, the judge has positional bias.
    """
    # Round 1: good answer first
    r1_good, _, _ = await _call_openai_judge(question, answer_good, ground_truth)
    r1_bad, _, _ = await _call_openai_judge(question, answer_bad, ground_truth)

    # Score difference in normal order
    diff_normal = r1_good["score"] - r1_bad["score"]

    # Round 2: swap order — present bad answer first, then judge good
    r2_bad, _, _ = await _call_openai_judge(question, answer_bad, ground_truth)
    r2_good, _, _ = await _call_openai_judge(question, answer_good, ground_truth)

    diff_swapped = r2_good["score"] - r2_bad["score"]

    bias_detected = abs(diff_normal - diff_swapped) > 1

    return {
        "bias_detected": bias_detected,
        "diff_normal_order": diff_normal,
        "diff_swapped_order": diff_swapped,
        "bias_magnitude": abs(diff_normal - diff_swapped),
        "interpretation": (
            "Positional bias detected — scoring changed when answer order was swapped."
            if bias_detected
            else "No significant positional bias detected."
        ),
    }


# ─── Main Judge class ─────────────────────────────────────────────────────────

class LLMJudge:
    """
    Multi-Judge Consensus Engine.
    Judges: GPT-4o-mini (Judge A) + Gemini 2.5 Flash (Judge B)

    Conflict resolution strategy:
      |diff| == 0  → Full agreement (1.0), use shared score
      |diff| == 1  → Soft agreement (0.5), use average
      |diff| >= 2  → Conflict (0.0), invoke tie-breaker meta-judge
    """

    CONFLICT_THRESHOLD = 1   # score difference where tie-breaker is triggered

    def __init__(self):
        self.rubrics = {
            "accuracy": "Score 1-5 on factual correctness compared to ground truth.",
            "completeness": "Score 1-5 on coverage of key points from ground truth.",
            "tone": "Score 1-5 on professionalism and appropriateness of language.",
        }
        # Accumulate scores for batch-level Cohen's Kappa
        self._batch_scores_a: List[int] = []
        self._batch_scores_b: List[int] = []

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single QA pair with both judges concurrently.
        Includes automatic conflict resolution.
        """
        # Run both judges in parallel
        (judge_a, _, _), (judge_b, _, _) = await asyncio.gather(
            _call_openai_judge(question, answer, ground_truth),
            _call_gemini_judge(question, answer, ground_truth),
        )

        score_a: int = judge_a["score"]
        score_b: int = judge_b["score"]
        diff = abs(score_a - score_b)

        # Track for batch Cohen's Kappa
        self._batch_scores_a.append(score_a)
        self._batch_scores_b.append(score_b)

        # Conflict resolution
        tiebreaker_result = None
        if diff == 0:
            agreement_rate = 1.0
            final_score = float(score_a)
        elif diff == 1:
            agreement_rate = 0.5
            final_score = (score_a + score_b) / 2.0
        else:
            # Hard conflict → invoke meta-judge
            agreement_rate = 0.0
            tiebreaker_result = await _call_tiebreaker(
                question=question, answer=answer, ground_truth=ground_truth,
                score_a=score_a, reason_a=judge_a["reason"],
                score_b=score_b, reason_b=judge_b["reason"],
            )
            final_score = float(tiebreaker_result["final_score"])

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "conflict_detected": diff > self.CONFLICT_THRESHOLD,
            "individual_scores": {
                judge_a["model"]: score_a,
                judge_b["model"]: score_b,
            },
            "reasoning": {
                judge_a["model"]: judge_a["reason"],
                judge_b["model"]: judge_b["reason"],
            },
            "tiebreaker": tiebreaker_result,
            "latency": {
                judge_a["model"]: judge_a.get("latency_ms"),
                judge_b["model"]: judge_b.get("latency_ms"),
            },
        }

    def compute_batch_kappa(self) -> float:
        """Compute Cohen's Kappa for all evaluations done in this judge session."""
        return compute_cohens_kappa(self._batch_scores_a, self._batch_scores_b)

    def reset_batch(self):
        """Reset batch accumulators for a new benchmark run."""
        self._batch_scores_a.clear()
        self._batch_scores_b.clear()

    @staticmethod
    def get_cost_summary(eval_count: int = None) -> Dict:
        """Return cost and token usage summary for the entire run."""
        return _cost_tracker.summary(eval_count=eval_count)

    @classmethod
    def reset_cost_tracker(cls):
        """Reset the global cost tracker before a new benchmark run."""
        global _cost_tracker
        _cost_tracker = CostTracker()
