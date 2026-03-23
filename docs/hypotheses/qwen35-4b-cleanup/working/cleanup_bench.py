#!/usr/bin/env python3
"""Benchmark script for Qwen3.5-4B STT cleanup hypothesis test.

Retest of qwen35-stt-cleanup (2B, disproven). Same test cases and system
prompt, updated model ID and thresholds for the larger 4B model.

Loads Qwen3.5-4B 8-bit via mlx-lm (or mlx-vlm if needed), runs cleanup
on all test cases, measures quality and latency, and writes results to JSON.

Usage:
    cd docs/hypotheses/qwen35-4b-cleanup/working
    .venv/bin/python cleanup_bench.py
"""
import json, os, re, sys, time
from datetime import datetime, timezone
import mlx.core as mx
from test_cases import TEST_CASES

# -- Model loading --
# Qwen3.5-4B is a unified VLM with early fusion. The mlx-community 8-bit
# variant may require mlx_vlm rather than mlx_lm. We try mlx_lm first
# since it's simpler for text-only use.
MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-8bit"
MAX_TOKENS = 512

# Same system prompt as the 2B test for fair comparison
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


def load_model():
    """Try mlx_lm first, fall back to mlx_vlm for text-only generation."""
    try:
        from mlx_lm import generate as mlx_lm_generate, load as mlx_lm_load
        print(f"Loading model via mlx_lm: {MODEL_ID}")
        model, tokenizer = mlx_lm_load(MODEL_ID)
        return model, tokenizer, mlx_lm_generate, "mlx_lm"
    except Exception as e:
        print(f"mlx_lm failed ({e}), trying mlx_vlm...")

    from mlx_vlm import generate as mlx_vlm_generate, load as mlx_vlm_load
    print(f"Loading model via mlx_vlm: {MODEL_ID}")
    model, processor = mlx_vlm_load(MODEL_ID)
    return model, processor, mlx_vlm_generate, "mlx_vlm"


def do_generate(model, tokenizer_or_processor, generate_fn, backend, prompt, max_tokens):
    """Unified generate call that works with both mlx_lm and mlx_vlm."""
    if backend == "mlx_lm":
        return generate_fn(model, tokenizer_or_processor, prompt,
                           max_tokens=max_tokens, verbose=False).strip()
    else:
        # mlx_vlm generate for text-only (no image)
        return generate_fn(model, tokenizer_or_processor, prompt,
                           max_tokens=max_tokens, verbose=False).strip()


def get_tokenizer(tokenizer_or_processor, backend):
    """Extract the tokenizer from processor if using mlx_vlm."""
    if backend == "mlx_lm":
        return tokenizer_or_processor
    else:
        # mlx_vlm processor has a tokenizer attribute
        if hasattr(tokenizer_or_processor, "tokenizer"):
            return tokenizer_or_processor.tokenizer
        return tokenizer_or_processor


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
_PUNCT_RE = re.compile(r"[,.\-!?;:\"'()]+")

def _normalize_for_filler_check(text: str) -> str:
    """Strip punctuation so filler words adjacent to commas/periods are detected."""
    return _PUNCT_RE.sub(" ", text.lower())

def check_filler_clean(output: str) -> bool:
    normalized = _normalize_for_filler_check(output)
    for f in FILLER_WORDS:
        if f == "like":
            padded = f" {normalized} "
            if " like " not in padded:
                continue
            cleaned = normalized
            for legit in _LIKE_LEGIT:
                cleaned = cleaned.replace(legit, "")
            if " like " not in f" {cleaned} ":
                continue
            return False
        elif f" {f} " in f" {normalized} ":
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

def check_echo_back(output: str, raw_input: str) -> bool:
    """Check if output is essentially the raw input echoed back unchanged.
    Returns True if the output is an echo (bad), False if the model actually edited."""
    return levenshtein_similarity(output.lower().strip(), raw_input.lower().strip()) > 0.95


def _percentile(vals: list, pct: float) -> float:
    if not vals: return 0.0
    idx = (pct / 100.0) * (len(vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])


def main() -> None:
    t0 = time.perf_counter()
    model, tok_or_proc, generate_fn, backend = load_model()
    load_time = time.perf_counter() - t0
    tokenizer = get_tokenizer(tok_or_proc, backend)
    print(f"Model loaded via {backend} in {load_time:.1f}s, "
          f"peak memory: {mx.get_peak_memory()/1e9:.2f} GB")

    # Warmup
    warmup_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello world"},
    ]
    warmup_prompt = tokenizer.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    _ = do_generate(model, tok_or_proc, generate_fn, backend,
                    warmup_prompt, max_tokens=32)
    print("Warmup complete.\n")

    # Run test cases
    results = []
    for tc in TEST_CASES:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tc["raw"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        input_tokens = len(tokenizer.encode(prompt))

        t_start = time.perf_counter()
        output = do_generate(model, tok_or_proc, generate_fn, backend,
                             prompt, max_tokens=MAX_TOKENS)
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
        echo = check_echo_back(output, tc["raw"])

        checks = [meaning_ok, halluc_ok, format_ok]
        if cat in ("filler", "ramble"): checks.append(filler_ok)
        if cat == "clean-passthrough": checks.append(pt_ok)
        passed = all(checks)

        notes = []
        if echo: notes.append("ECHO-BACK")
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
            "echo_back": echo, "passed": passed,
            "notes": "; ".join(notes),
            "raw_input": tc["raw"], "expected": exp, "actual_output": output,
        })

    # Print results table
    print("=" * 110)
    hdr = (f"{'Name':<30} {'Category':<18} {'Lat(ms)':>8} {'tok/s':>7} "
           f"{'MeanSim':>8} {'Echo':>5} {'Pass':>5} Notes")
    print(hdr)
    print("-" * 110)
    for r in results:
        print(f"{r['name']:<30} {r['category']:<18} {r['latency_ms']:>8.1f} "
              f"{r['tok_s']:>7.1f} {r['meaning_sim']:>8.3f} "
              f"{'YES' if r['echo_back'] else 'no':>5} "
              f"{'PASS' if r['passed'] else 'FAIL':>5} {r['notes']}")
    print("=" * 110)

    # Print actual outputs
    print("\n--- Actual outputs ---")
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        echo_tag = " [ECHO]" if r["echo_back"] else ""
        print(f"\n[{tag}]{echo_tag} {r['name']}:")
        for label, key in [("Raw", "raw_input"), ("Expected", "expected"),
                           ("Actual", "actual_output")]:
            v = r[key]
            print(f"  {label:>8}: {v[:90]}{'...' if len(v) > 90 else ''}")

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

    echo_count = sum(1 for r in results if r["echo_back"])

    all_lat = sorted(r["latency_ms"] for r in results)
    p50 = _percentile(all_lat, 50)
    p95 = _percentile(all_lat, 95)
    p99 = _percentile(all_lat, 99)

    short_lat = sorted(r["latency_ms"] for r in results
                       if len(r["raw_input"].split()) <= 30)
    sp95 = _percentile(short_lat, 95) if short_lat else None

    all_tps = [r["tok_s"] for r in results if r["tok_s"] > 0]
    mean_tps = sum(all_tps) / len(all_tps) if all_tps else 0

    peak_mem = mx.get_peak_memory() / 1e9

    typ_lat = [r["latency_ms"] for r in results
               if 15 <= len(r["raw_input"].split()) <= 30]
    typ_med = _percentile(sorted(typ_lat), 50) if typ_lat else 0

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Filler removal rate:       {fc}/{len(filler_cases)} ({fr:.0%})")
    print(f"Self-correction rate:      {co}/{len(corr_cases)} ({cr:.0%})")
    print(f"Meaning preservation rate: {mo}/{len(results)} ({mr:.0%})")
    print(f"Format compliance rate:    {fo}/{len(results)} ({fmr:.0%})")
    print(f"Passthrough accuracy:      {po}/{len(pt_cases)} ({pr_:.0%})")
    print(f"Echo-back count:           {echo_count}/{len(results)}")
    print(f"\nLatency (all):   p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
    if sp95 is not None:
        print(f"Latency (<=30w): p95={sp95:.0f}ms")
    else:
        print("Latency (<=30w): no cases")
    print(f"Token throughput: mean={mean_tps:.1f} tok/s")
    print(f"Peak memory: {peak_mem:.2f} GB")

    # -- Hypothesis criteria scoring (4B thresholds) --
    # Success criteria (relaxed from 2B where noted)
    s1 = fc >= 4                            # filler removal >= 4/5
    s2 = co >= 2                            # self-correction >= 2/2
    s3 = mo >= 10                           # meaning preservation >= 10/12
    s4 = fo >= 11                           # format compliance >= 11/12
    s5 = po >= 2                            # clean passthrough 2/2
    s6 = sp95 is not None and sp95 < 1500   # latency < 1500ms (was 800)
    s7 = mean_tps >= 30                     # throughput >= 30 tok/s (was 100)
    s8 = peak_mem < 6.0                     # memory < 6GB (was 4)

    # Failure criteria
    f1 = echo_count >= 4                    # echo-back persistence (NEW)
    mc = sum(1 for r in results if r["meaning_sim"] < 0.70)
    f2 = mc >= 3                            # meaning corruption
    f3 = typ_med >= 3000                    # unacceptable latency (was 1500)
    nc = sum(1 for r in results if not r["format_clean"])
    f4 = nc >= 3                            # instruction non-compliance
    f5 = (len(pt_cases) - po) >= 1          # quality regression on clean

    print(f"\n{'='*60}\nHYPOTHESIS CRITERIA SCORING\n{'='*60}")
    print(f"\n  --- Success Criteria ---")
    for desc, val, flag in [
        ("Filler removal >= 4/5", f"{fc}/{len(filler_cases)}", s1),
        ("Self-correction >= 2/2", f"{co}/{len(corr_cases)}", s2),
        ("Meaning preservation >= 10/12", f"{mo}/{len(results)}", s3),
        ("Format compliance >= 11/12", f"{fo}/{len(results)}", s4),
        ("Clean passthrough 2/2", f"{po}/{len(pt_cases)}", s5),
        ("Latency < 1500ms p95 (<=30w)", f"{sp95:.0f}ms" if sp95 else "N/A", s6),
        ("Decode speed >= 30 tok/s", f"{mean_tps:.1f} tok/s", s7),
        ("Memory < 6GB", f"{peak_mem:.2f} GB", s8),
    ]:
        st = "PASS" if flag else "FAIL"
        print(f"  [{st:>9}] {desc}: {val}")

    print(f"\n  --- Failure Criteria ---")
    for desc, val, flag in [
        ("Echo-back persistence >= 4/12", f"{echo_count}/{len(results)}", f1),
        ("Meaning corruption >= 3/12", f"{mc}/{len(results)}", f2),
        ("Latency >= 3s typical", f"{typ_med:.0f}ms median", f3),
        ("Instruction non-compliance >= 3/12", f"{nc}/{len(results)}", f4),
        ("Quality regression on clean >= 1/2", f"{len(pt_cases)-po}/{len(pt_cases)}", f5),
    ]:
        st = "TRIGGERED" if flag else "OK"
        print(f"  [{st:>9}] {desc}: {val}")

    # -- Comparison with 2B results --
    print(f"\n{'='*60}\n2B vs 4B COMPARISON\n{'='*60}")
    print(f"{'Metric':<35} {'2B Result':<20} {'4B Result':<20}")
    print("-" * 75)
    comparisons = [
        ("Filler removal", "2/5 (40%)", f"{fc}/{len(filler_cases)} ({fr:.0%})"),
        ("Self-correction", "0/2 (0%)", f"{co}/{len(corr_cases)} ({cr:.0%})"),
        ("Meaning preservation", "8/12 (67%)", f"{mo}/{len(results)} ({mr:.0%})"),
        ("Format compliance", "12/12 (100%)", f"{fo}/{len(results)} ({fmr:.0%})"),
        ("Passthrough accuracy", "2/2 (100%)", f"{po}/{len(pt_cases)} ({pr_:.0%})"),
        ("Echo-back count", "~6/12", f"{echo_count}/{len(results)}"),
        ("Latency p95 (<=30w)", "472ms", f"{sp95:.0f}ms" if sp95 else "N/A"),
        ("Mean tok/s", "49.1", f"{mean_tps:.1f}"),
        ("Peak memory", "2.55 GB", f"{peak_mem:.2f} GB"),
    ]
    for metric, v2b, v4b in comparisons:
        print(f"  {metric:<35} {v2b:<20} {v4b:<20}")

    # Write JSON results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reviews_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    json_path = os.path.join(reviews_dir, f"run-{ts}.json")
    summary = {
        "timestamp": ts,
        "model_id": MODEL_ID,
        "backend": backend,
        "model_load_time_s": round(load_time, 1),
        "peak_memory_gb": round(peak_mem, 2),
        "system_prompt": SYSTEM_PROMPT,
        "num_test_cases": len(results),
        "prior_test": "qwen35-stt-cleanup (2B, disproven)",
        "aggregate": {
            "filler_removal_rate": round(fr, 3),
            "self_correction_rate": round(cr, 3),
            "meaning_preservation_rate": round(mr, 3),
            "format_compliance_rate": round(fmr, 3),
            "passthrough_accuracy": round(pr_, 3),
            "echo_back_count": echo_count,
        },
        "latency": {
            "all_p50_ms": round(p50, 1),
            "all_p95_ms": round(p95, 1),
            "all_p99_ms": round(p99, 1),
            "short_p95_ms": round(sp95, 1) if sp95 else None,
            "typical_median_ms": round(typ_med, 1) if typ_lat else None,
        },
        "throughput": {"mean_tok_s": round(mean_tps, 1)},
        "hypothesis_criteria": {
            "success": {
                "filler_removal": {"required": ">=4/5", "actual": f"{fc}/{len(filler_cases)}", "pass": s1},
                "self_correction": {"required": ">=2/2", "actual": f"{co}/{len(corr_cases)}", "pass": s2},
                "meaning_preservation": {"required": ">=10/12", "actual": f"{mo}/{len(results)}", "pass": s3},
                "format_compliance": {"required": ">=11/12", "actual": f"{fo}/{len(results)}", "pass": s4},
                "clean_passthrough": {"required": "2/2", "actual": f"{po}/{len(pt_cases)}", "pass": s5},
                "latency_short": {"required": "<1500ms p95", "actual": f"{sp95:.0f}ms" if sp95 else "N/A", "pass": s6},
                "decode_speed": {"required": ">=30 tok/s", "actual": f"{mean_tps:.1f}", "pass": s7},
                "memory": {"required": "<6GB", "actual": f"{peak_mem:.2f}GB", "pass": s8},
            },
            "failure": {
                "echo_back_persistence": {"threshold": ">=4/12", "actual": f"{echo_count}/{len(results)}", "triggered": f1},
                "meaning_corruption": {"threshold": ">=3/12", "actual": f"{mc}/{len(results)}", "triggered": f2},
                "unacceptable_latency": {"threshold": ">=3s typical", "actual": f"{typ_med:.0f}ms", "triggered": f3},
                "instruction_noncompliance": {"threshold": ">=3/12", "actual": f"{nc}/{len(results)}", "triggered": f4},
                "quality_regression_clean": {"threshold": ">=1/2", "actual": f"{len(pt_cases)-po}/{len(pt_cases)}", "triggered": f5},
            },
        },
        "cases": [{k: r[k] for k in (
            "name", "category", "input_tokens", "output_tokens",
            "latency_ms", "tok_s", "meaning_sim", "filler_clean",
            "no_hallucination", "format_clean", "passthrough_ok",
            "echo_back", "passed", "notes",
            "raw_input", "expected", "actual_output",
        )} for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to: {json_path}")


if __name__ == "__main__":
    main()
