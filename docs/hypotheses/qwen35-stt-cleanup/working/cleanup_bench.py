#!/usr/bin/env python3
"""Benchmark script for Qwen3.5-2B STT cleanup hypothesis test.

Loads Qwen3.5-2B 8-bit via mlx-lm, runs cleanup on all test cases,
measures quality and latency, and writes results to JSON.

Usage:
    cd docs/hypotheses/qwen35-stt-cleanup/working
    .venv/bin/python cleanup_bench.py
"""
import json, os, time
from datetime import datetime, timezone
import mlx.core as mx
from mlx_lm import generate, load
from test_cases import TEST_CASES

MODEL_ID = "mlx-community/Qwen3.5-2B-8bit"
MAX_TOKENS = 512
SYSTEM_PROMPT = (
    "You are a speech-to-text cleanup tool. Your job is to transform raw "
    "voice transcriptions into clean, polished written text.\n\n"
    "Rules:\n"
    "1. Remove filler words: um, uh, like, you know, so, actually, basically\n"
    "2. Resolve self-corrections: keep only the final intended version\n"
    "3. Fix grammar, capitalization, and punctuation\n"
    "4. Preserve the speaker's meaning exactly -- never add content\n"
    "5. Output ONLY the cleaned text, no explanations, no markdown, no quotes\n"
    "6. If the input is already clean, return it unchanged"
)
FILLER_WORDS = {"um", "uh", "like", "you know", "so", "actually", "basically"}

# -- Levenshtein similarity (no external dependency) --
def _levenshtein_distance(s: str, t: str) -> int:
    n, m = len(s), len(t)
    if n == 0: return m
    if m == 0: return n
    prev, curr = list(range(m + 1)), [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]

def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b: return 1.0
    return 1.0 - _levenshtein_distance(a, b) / max(len(a), len(b))

# -- Quality checks --
_LIKE_LEGIT = {"seems like", "looks like", "would like", "feel like", "felt like",
               "sounds like", "something like", "more like", "just like",
               "acts like", "worked like"}
def check_filler_clean(output: str) -> bool:
    lower = output.lower()
    for f in FILLER_WORDS:
        if f == "like":
            # Only flag "like" when it is a discourse filler, not a preposition/verb
            padded = f" {lower} "
            if " like " not in padded:
                continue
            # Remove all legitimate uses, then check if "like" remains
            cleaned = lower
            for legit in _LIKE_LEGIT:
                cleaned = cleaned.replace(legit, "")
            if " like " not in f" {cleaned} ":
                continue
            return False
        elif f" {f} " in f" {lower} ":
            return False
    return True

def check_no_hallucination(output: str, expected: str) -> bool:
    out_w, exp_w = len(output.split()), len(expected.split())
    return out_w == 0 if exp_w == 0 else out_w <= exp_w * 1.3

def check_format_clean(output: str) -> bool:
    s = output.strip()
    if not s: return False
    if s.startswith(("I ", "I'")): return "```" not in s and "**" not in s
    if s.startswith(('"', "'", "`", "Here", "Sure")): return False
    return "```" not in s and "**" not in s

def check_passthrough(output: str, expected: str) -> bool:
    return output.rstrip(".!?,;:") == expected.rstrip(".!?,;:")

def _percentile(vals: list, pct: float) -> float:
    if not vals: return 0.0
    idx = (pct / 100.0) * (len(vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])

def main() -> None:
    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s, peak memory: {mx.get_peak_memory()/1e9:.2f} GB")

    # Warmup
    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Hello world"}],
        tokenize=False, add_generation_prompt=True)
    _ = generate(model, tokenizer, warmup_prompt, max_tokens=32)
    print("Warmup complete.\n")

    # Run test cases
    results = []
    for tc in TEST_CASES:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": tc["raw"]}],
            tokenize=False, add_generation_prompt=True)
        input_tokens = len(tokenizer.encode(prompt))
        t_start = time.perf_counter()
        output = generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, verbose=False).strip()
        latency_s = time.perf_counter() - t_start
        output_tokens = len(tokenizer.encode(output))
        latency_ms = latency_s * 1000
        tok_s = output_tokens / latency_s if latency_s > 0 else 0
        cat, exp = tc["category"], tc["expected"]
        filler_ok = check_filler_clean(output)
        meaning_sim = levenshtein_similarity(output.lower(), exp.lower())
        meaning_ok = meaning_sim >= 0.80
        halluc_ok = check_no_hallucination(output, exp)
        format_ok = check_format_clean(output)
        pt_ok = check_passthrough(output, exp) if cat == "clean-passthrough" else None
        checks = [meaning_ok, halluc_ok, format_ok]
        if cat in ("filler", "ramble"): checks.append(filler_ok)
        if cat == "clean-passthrough": checks.append(pt_ok)
        passed = all(checks)
        notes = []
        if not filler_ok and cat in ("filler", "ramble"): notes.append("fillers remain")
        if not meaning_ok: notes.append(f"meaning_sim={meaning_sim:.2f}")
        if not halluc_ok: notes.append("hallucination")
        if not format_ok: notes.append("bad format")
        if pt_ok is False: notes.append("passthrough changed")
        results.append({
            "name": tc["name"], "category": cat,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1), "tok_s": round(tok_s, 1),
            "meaning_sim": round(meaning_sim, 3),
            "filler_clean": filler_ok, "no_hallucination": halluc_ok,
            "format_clean": format_ok, "passthrough_ok": pt_ok,
            "passed": passed, "notes": "; ".join(notes),
            "raw_input": tc["raw"], "expected": exp, "actual_output": output,
        })

    # Print results table
    print("=" * 100)
    hdr = f"{'Name':<30} {'Category':<18} {'Lat(ms)':>8} {'tok/s':>7} {'MeanSim':>8} {'Pass':>5} Notes"
    print(hdr)
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<30} {r['category']:<18} {r['latency_ms']:>8.1f} "
              f"{r['tok_s']:>7.1f} {r['meaning_sim']:>8.3f} "
              f"{'PASS' if r['passed'] else 'FAIL':>5} {r['notes']}")
    print("=" * 100)

    # Print actual outputs
    print("\n--- Actual outputs ---")
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{tag}] {r['name']}:")
        for label, key in [("Raw", "raw_input"), ("Expected", "expected"), ("Actual", "actual_output")]:
            v = r[key]
            print(f"  {label:>8}: {v[:80]}{'...' if len(v) > 80 else ''}")

    # Aggregate stats
    filler_cases = [r for r in results if r["category"] in ("filler", "ramble")]
    fc = sum(1 for r in filler_cases if r["filler_clean"])
    fr = fc / len(filler_cases) if filler_cases else 0
    corr_cases = [r for r in results if r["category"] == "self-correction"]
    co = sum(1 for r in corr_cases if r["meaning_sim"] >= 0.80)
    cr = co / len(corr_cases) if corr_cases else 0
    mo = sum(1 for r in results if r["meaning_sim"] >= 0.80)
    mr = mo / len(results)
    fo = sum(1 for r in results if r["format_clean"])
    fmr = fo / len(results)
    pt_cases = [r for r in results if r["category"] == "clean-passthrough"]
    po = sum(1 for r in pt_cases if r["passthrough_ok"])
    pr_ = po / len(pt_cases) if pt_cases else 0
    all_lat = sorted(r["latency_ms"] for r in results)
    p50, p95, p99 = _percentile(all_lat, 50), _percentile(all_lat, 95), _percentile(all_lat, 99)
    short_lat = sorted(r["latency_ms"] for r in results if len(r["raw_input"].split()) <= 30)
    sp95 = _percentile(short_lat, 95) if short_lat else None
    all_tps = [r["tok_s"] for r in results if r["tok_s"] > 0]
    mean_tps = sum(all_tps) / len(all_tps) if all_tps else 0
    peak_mem = mx.get_peak_memory() / 1e9
    typ_lat = [r["latency_ms"] for r in results if 15 <= len(r["raw_input"].split()) <= 30]
    typ_med = _percentile(sorted(typ_lat), 50) if typ_lat else 0

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Filler removal rate:       {fc}/{len(filler_cases)} ({fr:.0%})")
    print(f"Self-correction rate:      {co}/{len(corr_cases)} ({cr:.0%})")
    print(f"Meaning preservation rate: {mo}/{len(results)} ({mr:.0%})")
    print(f"Format compliance rate:    {fo}/{len(results)} ({fmr:.0%})")
    print(f"Passthrough accuracy:      {po}/{len(pt_cases)} ({pr_:.0%})")
    print(f"\nLatency (all):   p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
    if sp95 is not None: print(f"Latency (<=30w): p95={sp95:.0f}ms")
    else: print("Latency (<=30w): no cases")
    print(f"Token throughput: mean={mean_tps:.1f} tok/s")
    print(f"Peak memory: {peak_mem:.2f} GB")

    # Hypothesis criteria scoring
    s1, s2, s3, s4 = fr >= 0.9, cr >= 0.8, fmr >= 0.9, mr >= 0.9
    s5 = sp95 is not None and sp95 < 800
    s6, s7 = mean_tps >= 100, peak_mem < 4.0
    mc = sum(1 for r in results if r["meaning_sim"] < 0.70)
    f1, f2 = mc >= 3, typ_med >= 1500
    nc = sum(1 for r in results if not r["format_clean"])
    f3, f4 = nc >= 3, (len(pt_cases) - po) >= 2

    print(f"\n{'='*60}\nHYPOTHESIS CRITERIA SCORING\n{'='*60}")
    for kind, desc, val, flag in [
        ("S", "Filler removal >= 9/10", f"{fc}/{len(filler_cases)}", s1),
        ("S", "Self-correction >= 8/10", f"{co}/{len(corr_cases)}", s2),
        ("S", "Grammar & punctuation >= 9/10", f"{fo}/{len(results)}", s3),
        ("S", "Meaning preservation >= 9/10", f"{mo}/{len(results)}", s4),
        ("S", "Latency < 800ms p95 (<=30w)", f"{sp95:.0f}ms" if sp95 else "N/A", s5),
        ("S", "Decode speed >= 100 tok/s", f"{mean_tps:.1f} tok/s", s6),
        ("S", "Memory < 4GB", f"{peak_mem:.2f} GB", s7),
        ("F", "Meaning corruption >= 3/10", f"{mc}/{len(results)}", f1),
        ("F", "Latency >= 1.5s typical", f"{typ_med:.0f}ms median", f2),
        ("F", "Instruction non-compliance >= 3/10", f"{nc}/{len(results)}", f3),
        ("F", "Quality regression on clean >= 2/10", f"{len(pt_cases)-po}/{len(pt_cases)}", f4),
    ]:
        st = ("PASS" if flag else "FAIL") if kind == "S" else ("TRIGGERED" if flag else "OK")
        print(f"  [{st:>9}] {desc}: {val}")

    # Write JSON results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reviews_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    json_path = os.path.join(reviews_dir, f"run-{ts}.json")
    summary = {
        "timestamp": ts, "model_id": MODEL_ID,
        "model_load_time_s": round(load_time, 1),
        "peak_memory_gb": round(peak_mem, 2), "system_prompt": SYSTEM_PROMPT,
        "num_test_cases": len(results),
        "aggregate": {"filler_removal_rate": round(fr, 3),
                      "self_correction_rate": round(cr, 3),
                      "meaning_preservation_rate": round(mr, 3),
                      "format_compliance_rate": round(fmr, 3),
                      "passthrough_accuracy": round(pr_, 3)},
        "latency": {"all_p50_ms": round(p50, 1), "all_p95_ms": round(p95, 1),
                    "all_p99_ms": round(p99, 1),
                    "short_p95_ms": round(sp95, 1) if sp95 else None,
                    "typical_median_ms": round(typ_med, 1) if typ_lat else None},
        "throughput": {"mean_tok_s": round(mean_tps, 1)},
        "hypothesis_criteria": {
            "success": {
                "filler_removal": {"required": ">=9/10", "actual": f"{fc}/{len(filler_cases)}", "pass": s1},
                "self_correction": {"required": ">=8/10", "actual": f"{co}/{len(corr_cases)}", "pass": s2},
                "grammar_punctuation": {"required": ">=9/10", "actual": f"{fo}/{len(results)}", "pass": s3},
                "meaning_preservation": {"required": ">=9/10", "actual": f"{mo}/{len(results)}", "pass": s4},
                "latency_short": {"required": "<800ms p95", "actual": f"{sp95:.0f}ms" if sp95 else "N/A", "pass": s5},
                "decode_speed": {"required": ">=100 tok/s", "actual": f"{mean_tps:.1f}", "pass": s6},
                "memory": {"required": "<4GB", "actual": f"{peak_mem:.2f}GB", "pass": s7}},
            "failure": {
                "meaning_corruption": {"threshold": ">=3/10", "actual": f"{mc}/{len(results)}", "triggered": f1},
                "unacceptable_latency": {"threshold": ">=1.5s typical", "actual": f"{typ_med:.0f}ms", "triggered": f2},
                "instruction_noncompliance": {"threshold": ">=3/10", "actual": f"{nc}/{len(results)}", "triggered": f3},
                "quality_regression_clean": {"threshold": ">=2/10", "actual": f"{len(pt_cases)-po}/{len(pt_cases)}", "triggered": f4}}},
        "cases": [{k: r[k] for k in ("name", "category", "input_tokens", "output_tokens",
                   "latency_ms", "tok_s", "meaning_sim", "filler_clean", "no_hallucination",
                   "format_clean", "passthrough_ok", "passed", "notes",
                   "raw_input", "expected", "actual_output")} for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to: {json_path}")

if __name__ == "__main__":
    main()
