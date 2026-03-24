"""CoEdit-large STT cleanup benchmark.

Evaluates grammarly/coedit-large on 12 speech-to-text cleanup test cases
across 4 instruction prompts. Measures quality (Levenshtein similarity,
filler removal, self-correction, passthrough, hallucination) and latency.

Usage:
    cd /Users/jud/Projects/operator
    .venv-poc/bin/python docs/hypotheses/coedit-cleanup/working/cleanup_bench.py
"""

import difflib
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from test_cases import TEST_CASES

# --- Configuration ---

MODEL_NAME = "grammarly/coedit-large"

PROMPTS = [
    "Make this text fluent: ",
    "Fix the grammar in this text: ",
    "Remove disfluencies from this text: ",
    "Fix the grammar and remove filler words: ",
]

FILLER_PATTERN = re.compile(
    r"\b(um|uh|like|you know|so like|yeah like)\b", re.IGNORECASE
)

FILLER_CATEGORIES = {"filler", "ramble"}
CORRECTION_CATEGORIES = {"self-correction"}
PASSTHROUGH_CATEGORIES = {"clean-passthrough"}


# --- Evaluation helpers ---

def levenshtein_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def count_fillers(text: str) -> int:
    """Count filler word occurrences in text."""
    return len(FILLER_PATTERN.findall(text))


def check_hallucination(raw: str, output: str) -> bool:
    """True if output has suspiciously more words than input (>1.5x)."""
    raw_wc = len(raw.split())
    out_wc = len(output.split())
    return out_wc > raw_wc * 1.5


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# --- Main ---

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reviews_dir = os.path.join(script_dir, "reviews")
    os.makedirs(reviews_dir, exist_ok=True)

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "mps"
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    print(f"Device: {device_name}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    # Run all prompt x case combinations
    all_results = {}

    for prompt in PROMPTS:
        prompt_results = []
        for case in TEST_CASES:
            input_text = prompt + case["raw"]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            t_start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    num_beams=5,
                    max_new_tokens=256,
                )
            latency_s = time.perf_counter() - t_start

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

            # Evaluate
            lev_sim = levenshtein_similarity(output_text, case["expected"])
            fillers_in = count_fillers(case["raw"])
            fillers_out = count_fillers(output_text)
            hallucinated = check_hallucination(case["raw"], output_text)

            # Category-specific checks
            category = case["category"]
            passed = True

            if category in FILLER_CATEGORIES:
                filler_ok = fillers_out < fillers_in
            else:
                filler_ok = None

            if category in PASSTHROUGH_CATEGORIES:
                passthrough_ok = lev_sim >= 0.95
                passed = passthrough_ok and not hallucinated
            elif category in CORRECTION_CATEGORIES:
                # For self-correction: lev_sim >= 0.75 to expected
                passed = lev_sim >= 0.75 and not hallucinated
            else:
                passed = lev_sim >= 0.75 and not hallucinated

            prompt_results.append({
                "name": case["name"],
                "category": category,
                "raw": case["raw"],
                "expected": case["expected"],
                "output": output_text,
                "lev_sim": round(lev_sim, 3),
                "latency_ms": round(latency_s * 1000, 1),
                "fillers_in": fillers_in,
                "fillers_out": fillers_out,
                "filler_ok": filler_ok,
                "hallucinated": hallucinated,
                "passed": passed,
            })

        all_results[prompt.strip()] = prompt_results

    # --- Print per-prompt results tables ---

    for prompt, results in all_results.items():
        print("=" * 120)
        print(f"PROMPT: \"{prompt}\"")
        print("=" * 120)
        header = f"{'name':<30} {'category':<20} {'output':<80} {'lev_sim':>8} {'lat_ms':>8} {'pass':>6}"
        print(header)
        print("-" * len(header))
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(
                f"{r['name']:<30} {r['category']:<20} "
                f"{truncate(r['output'], 80):<80} "
                f"{r['lev_sim']:>8.3f} {r['latency_ms']:>8.1f} {mark:>6}"
            )
        print()

    # --- Per-prompt summary ---

    print("=" * 120)
    print("PER-PROMPT SUMMARY")
    print("=" * 120)
    summary_hdr = (
        f"{'prompt':<50} {'filler':>8} {'correct':>8} {'passthru':>8} "
        f"{'meaning':>8} {'halluc':>8} {'avg_ms':>8}"
    )
    print(summary_hdr)
    print("-" * len(summary_hdr))

    best_prompt = None
    best_score = -1
    prompt_summaries = {}

    for prompt, results in all_results.items():
        filler_cases = [r for r in results if r["category"] in FILLER_CATEGORIES]
        filler_score = sum(1 for r in filler_cases if r["filler_ok"])
        filler_total = len(filler_cases)

        corr_cases = [r for r in results if r["category"] in CORRECTION_CATEGORIES]
        corr_score = sum(1 for r in corr_cases if r["lev_sim"] >= 0.75)
        corr_total = len(corr_cases)

        pass_cases = [r for r in results if r["category"] in PASSTHROUGH_CATEGORIES]
        pass_score = sum(1 for r in pass_cases if r["lev_sim"] >= 0.95)
        pass_total = len(pass_cases)

        meaning_score = sum(1 for r in results if r["lev_sim"] >= 0.75)
        meaning_total = len(results)

        halluc_count = sum(1 for r in results if r["hallucinated"])

        avg_lat = sum(r["latency_ms"] for r in results) / len(results)
        latencies = sorted(r["latency_ms"] for r in results)
        p95_lat = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]

        summary = {
            "filler": f"{filler_score}/{filler_total}",
            "correction": f"{corr_score}/{corr_total}",
            "passthrough": f"{pass_score}/{pass_total}",
            "meaning": f"{meaning_score}/{meaning_total}",
            "hallucination_count": halluc_count,
            "avg_latency_ms": round(avg_lat, 1),
            "p95_latency_ms": round(p95_lat, 1),
        }
        prompt_summaries[prompt] = summary

        print(
            f"{truncate(prompt, 50):<50} "
            f"{summary['filler']:>8} {summary['correction']:>8} "
            f"{summary['passthrough']:>8} {summary['meaning']:>8} "
            f"{halluc_count:>8} {avg_lat:>8.1f}"
        )

        # Score: filler + correction*3 + passthrough + meaning - halluc*2
        composite = filler_score + corr_score * 3 + pass_score + meaning_score - halluc_count * 2
        if composite > best_score:
            best_score = composite
            best_prompt = prompt

    print()
    print(f"BEST PROMPT: \"{best_prompt}\" (composite score: {best_score})")
    print()

    # --- Cross-model comparison ---

    print("=" * 120)
    print("CROSS-MODEL COMPARISON")
    print("=" * 120)
    comp_hdr = f"{'model':<40} {'filler':>8} {'correct':>8} {'meaning':>8} {'notes':<40}"
    print(comp_hdr)
    print("-" * len(comp_hdr))

    best_summary = prompt_summaries[best_prompt]
    print(
        f"{'CoEdit-large (best prompt)':<40} "
        f"{best_summary['filler']:>8} {best_summary['correction']:>8} "
        f"{best_summary['meaning']:>8} "
        f"{'p95 lat: ' + str(best_summary['p95_latency_ms']) + 'ms':<40}"
    )
    print(
        f"{'Qwen 2B (prior)':<40} {'2/5':>8} {'0/2':>8} {'--':>8} "
        f"{'echoed input verbatim, 0% correction':<40}"
    )
    print(
        f"{'Qwen 4B (prior)':<40} {'3/5':>8} {'0/2':>8} {'9/12':>8} "
        f"{'marginal improvement, 25 tok/s':<40}"
    )
    print(
        f"{'BERT disfluency (prior)':<40} {'--':>8} {'0/2':>8} {'--':>8} "
        f"{'13.2% FP rate, wrong domain':<40}"
    )
    print()

    # --- Success/failure criteria check ---

    print("=" * 120)
    print("HYPOTHESIS CRITERIA CHECK (best prompt)")
    print("=" * 120)

    best_results = all_results[best_prompt]

    filler_cases = [r for r in best_results if r["category"] in FILLER_CATEGORIES]
    filler_pass = sum(1 for r in filler_cases if r["filler_ok"])

    corr_cases = [r for r in best_results if r["category"] in CORRECTION_CATEGORIES]
    corr_pass = sum(1 for r in corr_cases if r["lev_sim"] >= 0.75)

    pass_cases = [r for r in best_results if r["category"] in PASSTHROUGH_CATEGORIES]
    pass_pass = sum(1 for r in pass_cases if r["lev_sim"] >= 0.95)

    meaning_pass = sum(1 for r in best_results if r["lev_sim"] >= 0.75)

    halluc_count = sum(1 for r in best_results if r["hallucinated"])

    latencies = sorted(r["latency_ms"] for r in best_results)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95_lat = latencies[p95_idx]

    criteria = [
        ("Filler removal >= 4/5", filler_pass >= 4, f"{filler_pass}/5"),
        ("Self-correction >= 1/2", corr_pass >= 1, f"{corr_pass}/2"),
        ("Meaning preservation >= 8/12", meaning_pass >= 8, f"{meaning_pass}/12"),
        ("Clean passthrough 2/2", pass_pass >= 2, f"{pass_pass}/2"),
        ("No hallucination", halluc_count == 0, f"{halluc_count} hallucinations"),
        ("Latency < 1500ms p95", p95_lat < 1500, f"p95={p95_lat:.0f}ms"),
    ]

    all_pass = True
    for label, ok, detail in criteria:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label} -- {detail}")

    print()
    if all_pass:
        print("VERDICT: ALL CRITERIA MET -- hypothesis SUPPORTED")
    else:
        print("VERDICT: NOT ALL CRITERIA MET -- see details above")
    print()

    # --- Write JSON results ---

    output_path = os.path.join(reviews_dir, "coedit-large-results.json")
    json_output = {
        "model": MODEL_NAME,
        "device": device_name,
        "load_time_s": round(load_time, 1),
        "prompts": {},
        "summaries": prompt_summaries,
        "best_prompt": best_prompt,
        "criteria": {label: {"passed": ok, "detail": detail} for label, ok, detail in criteria},
    }
    for prompt, results in all_results.items():
        json_output["prompts"][prompt] = results

    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
