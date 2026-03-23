#!/usr/bin/env python3
"""Benchmark script for BERT disfluency token classification hypothesis test.

Loads 4i-ai/BERT_disfluency_cls (BERT-base fine-tuned for disfluency detection),
runs token classification on the same 12 test cases from the Qwen3.5 tests,
inspects per-token labels, strips disfluent tokens, and evaluates quality/latency.

Usage:
    cd docs/hypotheses/bert-disfluency/working
    python classify_bench.py
"""
import json, os, re, sys, time, traceback
from datetime import datetime, timezone
from test_cases import TEST_CASES

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# -- Config --
MODEL_ID = "4i-ai/BERT_disfluency_cls"


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


# -- Filler check (same as Qwen benchmarks) --
FILLER_WORDS = {"um", "uh", "like", "you know", "so", "actually", "basically"}
_LIKE_LEGIT = {"seems like", "looks like", "would like", "feel like", "felt like",
               "sounds like", "something like", "more like", "just like",
               "acts like", "worked like"}
_PUNCT_RE = re.compile(r"[,.\-!?;:\"'()]+")

def _normalize_for_filler_check(text: str) -> str:
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


def check_passthrough(output: str, expected: str) -> bool:
    return output.rstrip(".!?,;:") == expected.rstrip(".!?,;:")


def load_model():
    """Load the BERT disfluency classifier from HuggingFace."""
    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    model.eval()
    load_time = time.perf_counter() - t0

    # Inspect label mapping
    id2label = model.config.id2label
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Label mapping: {id2label}")
    print(f"Num labels: {model.config.num_labels}")

    # Try to use MPS (Apple Silicon GPU) if available, fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model = model.to(device)

    return model, tokenizer, device, id2label, load_time


def classify_tokens(model, tokenizer, device, id2label, text):
    """Run token classification and return per-token labels.

    Returns:
        tokens: list of str (WordPiece tokens, excluding [CLS]/[SEP])
        labels: list of str (predicted label for each token)
        word_ids: list of int|None (word index for each token)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]  # (seq_len, 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]  # (seq_len, num_labels)
    predictions = torch.argmax(logits, dim=-1).cpu().tolist()

    # Decode tokens and align with predictions
    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Build result, skipping [CLS] and [SEP]
    result_tokens = []
    result_labels = []
    result_offsets = []
    for i, (tok, pred, offset) in enumerate(zip(tokens, predictions, offset_mapping)):
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>"):
            continue
        label = id2label.get(pred, f"UNKNOWN_{pred}")
        result_tokens.append(tok)
        result_labels.append(label)
        result_offsets.append((offset[0].item(), offset[1].item()))

    return result_tokens, result_labels, result_offsets


def reassemble_text(tokens, labels, original_text):
    """Strip disfluent tokens and reassemble into clean text.

    Uses a simple approach: keep tokens labeled as fluent, reconstruct by
    checking if the token is a WordPiece continuation (starts with ##) or
    a new word.
    """
    # Identify which label means "disfluent" -- the model may use various
    # naming conventions. We check for common patterns.
    disfluent_labels = set()
    fluent_labels = set()
    all_labels = set(labels)

    for label in all_labels:
        lower = label.lower()
        if any(x in lower for x in ["disf", "remove", "delete", "1", "bad", "noisy"]):
            disfluent_labels.add(label)
        elif any(x in lower for x in ["fluent", "keep", "0", "clean", "good"]):
            fluent_labels.add(label)

    # If we couldn't determine automatically, check if it's a simple 0/1 scheme
    if not disfluent_labels and not fluent_labels:
        # Try numeric: some models use LABEL_0 = fluent, LABEL_1 = disfluent
        # or vice versa. We'll need to inspect the model config.
        # For now, assume LABEL_1 = disfluent (common convention)
        for label in all_labels:
            if label in ("LABEL_1", "1", "D", "d"):
                disfluent_labels.add(label)
            else:
                fluent_labels.add(label)

    # If still ambiguous, report and treat all as fluent (conservative)
    if not disfluent_labels:
        print(f"  WARNING: Could not determine disfluent labels from: {all_labels}")
        print(f"  Treating all tokens as fluent (no removal)")
        return original_text, disfluent_labels, fluent_labels

    kept_pieces = []
    for tok, label in zip(tokens, labels):
        if label in disfluent_labels:
            continue
        if tok.startswith("##"):
            # WordPiece continuation -- append without space
            kept_pieces.append(tok[2:])
        else:
            if kept_pieces:
                kept_pieces.append(" ")
            kept_pieces.append(tok)

    result = "".join(kept_pieces).strip()

    # Basic cleanup: fix double spaces, orphaned punctuation
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([,.\-!?;:])", r"\1", result)
    result = re.sub(r"([,.\-!?;:])\s*([,.\-!?;:])", r"\1", result)

    return result, disfluent_labels, fluent_labels


def compute_false_positive_rate(tokens, labels, disfluent_labels):
    """Compute false positive rate: content words incorrectly tagged as disfluent.

    We define "content words" as tokens that are NOT known fillers and NOT
    WordPiece continuations of fillers. This is approximate since we don't
    have ground-truth token labels.
    """
    filler_stems = {"um", "uh", "like", "you", "know", "so", "actually",
                    "basically", "well", "yeah", "er", "hmm", "hm"}
    total_content = 0
    false_positives = 0

    for tok, label in zip(tokens, labels):
        clean_tok = tok.lstrip("#").lower()
        if clean_tok in filler_stems:
            continue  # skip fillers (not content)
        total_content += 1
        if label in disfluent_labels:
            false_positives += 1

    fp_rate = false_positives / total_content if total_content > 0 else 0.0
    return fp_rate, false_positives, total_content


def main() -> None:
    model, tokenizer, device, id2label, load_time = load_model()

    # Warmup
    print("Running warmup inference...")
    _ = classify_tokens(model, tokenizer, device, id2label, "Hello world, this is a test.")
    print("Warmup complete.\n")

    results = []
    all_fp_total = 0
    all_content_total = 0

    for tc in TEST_CASES:
        name = tc["name"]
        raw = tc["raw"]
        expected = tc["expected"]
        category = tc["category"]

        # Run classification
        t_start = time.perf_counter()
        tokens, labels, offsets = classify_tokens(model, tokenizer, device, id2label, raw)
        latency_s = time.perf_counter() - t_start
        latency_ms = latency_s * 1000

        # Reassemble
        cleaned, disfluent_labels, fluent_labels = reassemble_text(tokens, labels, raw)

        # Quality metrics
        meaning_sim = levenshtein_similarity(cleaned.lower(), expected.lower())
        filler_ok = check_filler_clean(cleaned)
        pt_ok = check_passthrough(cleaned, expected) if category == "clean-passthrough" else None

        # Token-level stats
        total_tokens = len(tokens)
        disfluent_count = sum(1 for l in labels if l in disfluent_labels)
        fluent_count = total_tokens - disfluent_count
        disf_rate = disfluent_count / total_tokens if total_tokens > 0 else 0

        # False positive analysis
        fp_rate, fp_count, content_count = compute_false_positive_rate(
            tokens, labels, disfluent_labels)
        all_fp_total += fp_count
        all_content_total += content_count

        # Build per-token label grid for inspection
        token_grid = []
        for tok, label in zip(tokens, labels):
            tag = "D" if label in disfluent_labels else "F"
            token_grid.append(f"{tok}[{tag}]")

        # Determine pass/fail
        checks = [meaning_sim >= 0.75]
        if category in ("filler", "ramble"):
            checks.append(filler_ok)
        if category == "clean-passthrough":
            checks.append(pt_ok)
        passed = all(checks)

        notes = []
        if not filler_ok and category in ("filler", "ramble"):
            notes.append("fillers remain")
        if meaning_sim < 0.75:
            notes.append(f"meaning_sim={meaning_sim:.2f}")
        if pt_ok is False:
            notes.append("passthrough changed")
        if fp_rate > 0.05:
            notes.append(f"high FP rate={fp_rate:.1%}")

        results.append({
            "name": name,
            "category": category,
            "latency_ms": round(latency_ms, 2),
            "total_tokens": total_tokens,
            "disfluent_count": disfluent_count,
            "fluent_count": fluent_count,
            "disf_rate": round(disf_rate, 3),
            "meaning_sim": round(meaning_sim, 3),
            "filler_clean": filler_ok,
            "passthrough_ok": pt_ok,
            "fp_rate": round(fp_rate, 4),
            "fp_count": fp_count,
            "content_count": content_count,
            "passed": passed,
            "notes": "; ".join(notes),
            "raw_input": raw,
            "expected": expected,
            "actual_output": cleaned,
            "token_grid": " ".join(token_grid),
            "disfluent_labels_used": sorted(disfluent_labels),
            "fluent_labels_used": sorted(fluent_labels),
        })

    # -- Print results --
    print("=" * 120)
    hdr = (f"{'Name':<30} {'Category':<18} {'Lat(ms)':>8} {'Tokens':>7} "
           f"{'Disf':>5} {'MeanSim':>8} {'FPRate':>7} {'Pass':>5} Notes")
    print(hdr)
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<30} {r['category']:<18} {r['latency_ms']:>8.2f} "
              f"{r['total_tokens']:>7} {r['disfluent_count']:>5} "
              f"{r['meaning_sim']:>8.3f} {r['fp_rate']:>7.1%} "
              f"{'PASS' if r['passed'] else 'FAIL':>5} {r['notes']}")
    print("=" * 120)

    # -- Per-token label grids --
    print("\n--- Per-Token Label Grids ---")
    print("(F=fluent, D=disfluent)\n")
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"[{tag}] {r['name']} ({r['category']}):")
        # Wrap token grid at ~100 chars
        grid = r["token_grid"]
        lines = []
        line = "  "
        for piece in grid.split(" "):
            if len(line) + len(piece) + 1 > 100:
                lines.append(line)
                line = "  "
            line += piece + " "
        if line.strip():
            lines.append(line)
        for l in lines:
            print(l.rstrip())
        print(f"  -> Cleaned: {r['actual_output'][:100]}{'...' if len(r['actual_output']) > 100 else ''}")
        print(f"  -> Expected: {r['expected'][:100]}{'...' if len(r['expected']) > 100 else ''}")
        print()

    # -- Aggregate stats --
    filler_cases = [r for r in results if r["category"] in ("filler", "ramble")]
    fc = sum(1 for r in filler_cases if r["filler_clean"])
    fr = fc / len(filler_cases) if filler_cases else 0

    corr_cases = [r for r in results if r["category"] == "self-correction"]
    co = sum(1 for r in corr_cases if r["meaning_sim"] >= 0.75)
    cr = co / len(corr_cases) if corr_cases else 0

    pt_cases = [r for r in results if r["category"] == "clean-passthrough"]
    po = sum(1 for r in pt_cases if r["passthrough_ok"])
    pr_ = po / len(pt_cases) if pt_cases else 0

    mo = sum(1 for r in results if r["meaning_sim"] >= 0.75)
    mr = mo / len(results)

    global_fp = all_fp_total / all_content_total if all_content_total > 0 else 0

    all_lat = sorted(r["latency_ms"] for r in results)
    def _pct(vals, pct):
        if not vals: return 0.0
        idx = (pct / 100.0) * (len(vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
        return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])

    p50 = _pct(all_lat, 50)
    p95 = _pct(all_lat, 95)
    p99 = _pct(all_lat, 99)
    max_lat = max(all_lat) if all_lat else 0

    # Memory estimate (model params)
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / 1e6  # float32, 4 bytes each
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Parameters: {param_count:,} ({param_mb:.0f} MB at fp32)")
    print(f"Load time: {load_time:.2f}s")
    print(f"Label mapping: {id2label}")
    print(f"\nFiller removal rate:       {fc}/{len(filler_cases)} ({fr:.0%})")
    print(f"Self-correction rate:      {co}/{len(corr_cases)} ({cr:.0%})")
    print(f"Meaning preservation:      {mo}/{len(results)} ({mr:.0%})")
    print(f"Passthrough accuracy:      {po}/{len(pt_cases)} ({pr_:.0%})")
    print(f"Global FP rate:            {all_fp_total}/{all_content_total} ({global_fp:.1%})")
    print(f"\nLatency: p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  max={max_lat:.1f}ms")

    # -- Hypothesis criteria scoring --
    s1 = fc >= 4                            # filler detection >= 4/5
    s2 = co >= 1                            # self-correction >= 1/2
    s3 = global_fp < 0.05                   # content preservation FP < 5%
    s4 = po >= 2                            # clean passthrough 2/2
    s5 = mo >= 8                            # reassembly quality >= 8/12
    s6 = p95 < 50                           # latency < 50ms p95
    s7 = param_mb < 1000                    # memory < 1GB

    f1 = global_fp >= 0.10                  # massive false positives
    f2 = fc <= 2                            # filler blindness (<=2/5, worse than regex)
    f3 = co == 0                            # complete self-correction failure
    f4 = p95 >= 200                         # unacceptable latency

    print(f"\n{'='*60}\nHYPOTHESIS CRITERIA SCORING\n{'='*60}")
    print(f"\n  --- Success Criteria ---")
    for desc, val, flag in [
        ("Filler detection >= 4/5", f"{fc}/{len(filler_cases)}", s1),
        ("Self-correction detection >= 1/2", f"{co}/{len(corr_cases)}", s2),
        ("Content preservation FP < 5%", f"{global_fp:.1%}", s3),
        ("Clean passthrough 2/2", f"{po}/{len(pt_cases)}", s4),
        ("Reassembly quality >= 8/12", f"{mo}/{len(results)}", s5),
        ("Latency < 50ms p95", f"{p95:.1f}ms", s6),
        ("Memory < 1GB", f"{param_mb:.0f}MB", s7),
    ]:
        st = "PASS" if flag else "FAIL"
        print(f"  [{st:>9}] {desc}: {val}")

    print(f"\n  --- Failure Criteria ---")
    for desc, val, flag in [
        ("Massive FP >= 10% content tokens", f"{global_fp:.1%}", f1),
        ("Filler blindness <= 2/5", f"{fc}/{len(filler_cases)}", f2),
        ("Complete self-correction failure 0/2", f"{co}/{len(corr_cases)}", f3),
        ("Unacceptable latency >= 200ms p95", f"{p95:.1f}ms", f4),
    ]:
        st = "TRIGGERED" if flag else "OK"
        print(f"  [{st:>9}] {desc}: {val}")

    # -- Comparison with Qwen results --
    print(f"\n{'='*60}\nCOMPARISON WITH PRIOR HYPOTHESES\n{'='*60}")
    print(f"{'Metric':<35} {'Qwen 2B':<15} {'Qwen 4B':<15} {'BERT cls':<15}")
    print("-" * 80)
    comparisons = [
        ("Filler removal", "2/5 (40%)", "3/5 (60%)", f"{fc}/{len(filler_cases)} ({fr:.0%})"),
        ("Self-correction", "0/2 (0%)", "0/2 (0%)", f"{co}/{len(corr_cases)} ({cr:.0%})"),
        ("Meaning preservation", "8/12 (67%)", "9/12 (75%)", f"{mo}/{len(results)} ({mr:.0%})"),
        ("Passthrough", "2/2 (100%)", "2/2 (100%)", f"{po}/{len(pt_cases)} ({pr_:.0%})"),
        ("Latency p95", "472ms", "~900ms", f"{p95:.0f}ms"),
        ("Memory", "2.55 GB", "~5 GB", f"{param_mb:.0f}MB"),
    ]
    for metric, q2, q4, bert in comparisons:
        print(f"  {metric:<35} {q2:<15} {q4:<15} {bert:<15}")

    # -- Write JSON results --
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reviews_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    json_path = os.path.join(reviews_dir, f"run-{ts}.json")
    summary = {
        "timestamp": ts,
        "model_id": MODEL_ID,
        "device": str(device),
        "param_count": param_count,
        "param_mb": round(param_mb, 1),
        "load_time_s": round(load_time, 2),
        "label_mapping": {str(k): v for k, v in id2label.items()},
        "num_test_cases": len(results),
        "prior_tests": [
            "qwen35-stt-cleanup (2B, disproven)",
            "qwen35-4b-cleanup (4B, disproven)"
        ],
        "aggregate": {
            "filler_removal_rate": round(fr, 3),
            "self_correction_rate": round(cr, 3),
            "meaning_preservation_rate": round(mr, 3),
            "passthrough_accuracy": round(pr_, 3),
            "global_fp_rate": round(global_fp, 4),
            "global_fp_count": all_fp_total,
            "global_content_count": all_content_total,
        },
        "latency": {
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "max_ms": round(max_lat, 2),
        },
        "hypothesis_criteria": {
            "success": {
                "filler_detection": {"required": ">=4/5", "actual": f"{fc}/{len(filler_cases)}", "pass": s1},
                "self_correction": {"required": ">=1/2", "actual": f"{co}/{len(corr_cases)}", "pass": s2},
                "content_preservation": {"required": "FP<5%", "actual": f"{global_fp:.1%}", "pass": s3},
                "clean_passthrough": {"required": "2/2", "actual": f"{po}/{len(pt_cases)}", "pass": s4},
                "reassembly_quality": {"required": ">=8/12", "actual": f"{mo}/{len(results)}", "pass": s5},
                "latency": {"required": "<50ms p95", "actual": f"{p95:.1f}ms", "pass": s6},
                "memory": {"required": "<1GB", "actual": f"{param_mb:.0f}MB", "pass": s7},
            },
            "failure": {
                "massive_fp": {"threshold": ">=10%", "actual": f"{global_fp:.1%}", "triggered": f1},
                "filler_blindness": {"threshold": "<=2/5", "actual": f"{fc}/{len(filler_cases)}", "triggered": f2},
                "self_correction_failure": {"threshold": "0/2", "actual": f"{co}/{len(corr_cases)}", "triggered": f3},
                "unacceptable_latency": {"threshold": ">=200ms p95", "actual": f"{p95:.1f}ms", "triggered": f4},
            },
        },
        "cases": [{k: r[k] for k in (
            "name", "category", "latency_ms", "total_tokens",
            "disfluent_count", "fluent_count", "disf_rate",
            "meaning_sim", "filler_clean", "passthrough_ok",
            "fp_rate", "fp_count", "content_count",
            "passed", "notes",
            "raw_input", "expected", "actual_output", "token_grid",
            "disfluent_labels_used", "fluent_labels_used",
        )} for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to: {json_path}")


if __name__ == "__main__":
    main()
