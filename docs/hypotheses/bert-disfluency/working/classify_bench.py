#!/usr/bin/env python3
"""Benchmark script for ModernBERT disfluency token classification hypothesis test.

Loads two ModernBERT token classifiers:
  - arielcerdap/modernbert-base-multiclass-disfluency-v2 (149M, base)
  - arielcerdap/modernbert-disfluency-expC-large-realonly (400M, large)

Both use 5-class labels: O (fluent), FP (filled pause), RP (repetition),
RV (revision), PW (partial word).

Runs token classification on the same 12 test cases from the Qwen3.5 tests,
inspects per-token labels, strips disfluent tokens, and evaluates quality/latency.

Usage:
    cd docs/hypotheses/bert-disfluency/working
    .venv/bin/python classify_bench.py
"""
import json, os, re, sys, time, traceback
from datetime import datetime, timezone
from test_cases import TEST_CASES

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# -- Config --
MODELS = [
    ("base", "arielcerdap/modernbert-base-multiclass-disfluency-v2"),
    ("large", "arielcerdap/modernbert-disfluency-expC-large-realonly"),
]

# 5-class label scheme: O is fluent, everything else is disfluent
DISFLUENT_LABELS = {"FP", "RP", "RV", "PW"}
FLUENT_LABELS = {"O"}


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


def load_model(model_id):
    """Load a ModernBERT disfluency classifier from HuggingFace."""
    print(f"Loading model: {model_id}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()
    load_time = time.perf_counter() - t0

    # Inspect label mapping
    id2label = model.config.id2label
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  Label mapping: {id2label}")
    print(f"  Num labels: {model.config.num_labels}")

    # Try to use MPS (Apple Silicon GPU) if available, fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("  Using CPU")
    model = model.to(device)

    return model, tokenizer, device, id2label, load_time


def classify_tokens(model, tokenizer, device, id2label, text):
    """Run token classification and return per-word labels using offset mapping.

    Uses offset_mapping to map subword predictions back to original words.
    Takes the label of the FIRST subtoken for each word.

    Returns:
        words: list of str (original words from the input text)
        labels: list of str (predicted label for each word, first-subtoken)
        offsets: list of (int, int) (character offsets per word in original text)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]  # (seq_len, 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]  # (seq_len, num_labels)
    predictions = torch.argmax(logits, dim=-1).cpu().tolist()

    # Use word_ids() to group subtokens into words and take first-subtoken label
    encoding = tokenizer(text, return_offsets_mapping=True)
    word_ids_list = encoding.word_ids()  # None for special tokens, int for words

    # Group by word_id: collect (label, start_offset, end_offset) per word
    word_data = {}  # word_id -> (first_subtoken_label, min_start, max_end)
    for idx, word_id in enumerate(word_ids_list):
        if word_id is None:
            continue  # skip special tokens ([CLS], [SEP])
        start, end = offset_mapping[idx].tolist()
        if start == end:
            continue  # skip empty spans
        label = id2label.get(predictions[idx], f"UNKNOWN_{predictions[idx]}")
        if word_id not in word_data:
            # First subtoken for this word -- use its label
            word_data[word_id] = (label, start, end)
        else:
            # Subsequent subtokens -- extend the span but keep first label
            _, prev_start, prev_end = word_data[word_id]
            word_data[word_id] = (word_data[word_id][0], prev_start, max(prev_end, end))

    # Sort by word_id to preserve order, extract word text from original
    words = []
    labels = []
    offsets = []
    for wid in sorted(word_data.keys()):
        label, start, end = word_data[wid]
        word_text = text[start:end]
        words.append(word_text)
        labels.append(label)
        offsets.append((start, end))

    return words, labels, offsets


def reassemble_text(tokens, labels, offsets, original_text):
    """Strip disfluent tokens using character offsets from original text."""
    kept_spans = []
    for label, (start, end) in zip(labels, offsets):
        if label in DISFLUENT_LABELS:
            continue
        if start == end:  # skip empty spans (special tokens)
            continue
        kept_spans.append((start, end))

    # Merge adjacent/overlapping spans and extract from original
    if not kept_spans:
        return ""
    pieces = []
    for start, end in kept_spans:
        pieces.append(original_text[start:end])
    result = " ".join(pieces)

    # Cleanup: collapse whitespace, fix orphaned punctuation
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([,.\-!?;:])", r"\1", result)
    return result.strip()


def compute_false_positive_rate(tokens, labels):
    """Compute false positive rate: content words incorrectly tagged as disfluent.

    We define "content words" as tokens that are NOT known fillers and NOT
    subword continuations of fillers. This is approximate since we don't
    have ground-truth token labels.
    """
    filler_stems = {"um", "uh", "like", "you", "know", "so", "actually",
                    "basically", "well", "yeah", "er", "hmm", "hm"}
    total_content = 0
    false_positives = 0

    for tok, label in zip(tokens, labels):
        clean_tok = tok.lower().strip()
        if clean_tok in filler_stems:
            continue  # skip fillers (not content)
        total_content += 1
        if label in DISFLUENT_LABELS:
            false_positives += 1

    fp_rate = false_positives / total_content if total_content > 0 else 0.0
    return fp_rate, false_positives, total_content


def run_model(model_name, model_id):
    """Run all test cases on a single model and return results."""
    model, tokenizer, device, id2label, load_time = load_model(model_id)

    # Warmup
    print(f"  Running warmup inference...")
    _ = classify_tokens(model, tokenizer, device, id2label, "Hello world, this is a test.")
    print(f"  Warmup complete.\n")

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
        words, labels, offsets = classify_tokens(model, tokenizer, device, id2label, raw)
        latency_s = time.perf_counter() - t_start
        latency_ms = latency_s * 1000

        # Reassemble
        cleaned = reassemble_text(words, labels, offsets, raw)

        # Quality metrics
        meaning_sim = levenshtein_similarity(cleaned.lower(), expected.lower())
        filler_ok = check_filler_clean(cleaned)
        pt_ok = check_passthrough(cleaned, expected) if category == "clean-passthrough" else None

        # Token-level stats
        total_tokens = len(words)
        disfluent_count = sum(1 for l in labels if l in DISFLUENT_LABELS)
        fluent_count = total_tokens - disfluent_count
        disf_rate = disfluent_count / total_tokens if total_tokens > 0 else 0

        # False positive analysis
        fp_rate, fp_count, content_count = compute_false_positive_rate(words, labels)
        all_fp_total += fp_count
        all_content_total += content_count

        # Build per-word label grid for inspection (shows actual 5-class labels)
        token_grid = []
        for word, label in zip(words, labels):
            token_grid.append(f"{word}[{label}]")

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
        })

    # Memory estimate (model params)
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / 1e6  # float32, 4 bytes each

    return {
        "model_name": model_name,
        "model_id": model_id,
        "device": str(device),
        "param_count": param_count,
        "param_mb": round(param_mb, 1),
        "load_time_s": round(load_time, 2),
        "label_mapping": {str(k): v for k, v in id2label.items()},
        "results": results,
        "all_fp_total": all_fp_total,
        "all_content_total": all_content_total,
    }


def print_model_results(model_run):
    """Print results table and per-token grids for a single model."""
    model_name = model_run["model_name"]
    model_id = model_run["model_id"]
    results = model_run["results"]

    print(f"\n{'=' * 120}")
    print(f"MODEL: {model_name} ({model_id})")
    print(f"{'=' * 120}")

    hdr = (f"{'Name':<30} {'Category':<18} {'Lat(ms)':>8} {'Words':>6} "
           f"{'Disf':>5} {'MeanSim':>8} {'FPRate':>7} {'Pass':>5} Notes")
    print(hdr)
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<30} {r['category']:<18} {r['latency_ms']:>8.2f} "
              f"{r['total_tokens']:>6} {r['disfluent_count']:>5} "
              f"{r['meaning_sim']:>8.3f} {r['fp_rate']:>7.1%} "
              f"{'PASS' if r['passed'] else 'FAIL':>5} {r['notes']}")
    print("-" * 120)

    # Per-word label grids
    print(f"\n--- Per-Word Label Grids [{model_name}] ---")
    print("(O=fluent, FP=filled pause, RP=repetition, RV=revision, PW=partial word)\n")
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
        for ln in lines:
            print(ln.rstrip())
        print(f"  -> Cleaned: {r['actual_output'][:100]}{'...' if len(r['actual_output']) > 100 else ''}")
        print(f"  -> Expected: {r['expected'][:100]}{'...' if len(r['expected']) > 100 else ''}")
        print()


def compute_aggregate_stats(model_run):
    """Compute aggregate stats for a model run."""
    results = model_run["results"]
    all_fp_total = model_run["all_fp_total"]
    all_content_total = model_run["all_content_total"]
    param_mb = model_run["param_mb"]

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

    return {
        "filler_cases": len(filler_cases), "filler_clean": fc, "filler_rate": fr,
        "corr_cases": len(corr_cases), "corr_clean": co, "corr_rate": cr,
        "pt_cases": len(pt_cases), "pt_clean": po, "pt_rate": pr_,
        "meaning_ok": mo, "meaning_rate": mr, "total_cases": len(results),
        "global_fp": global_fp, "global_fp_count": all_fp_total,
        "global_content_count": all_content_total,
        "p50": p50, "p95": p95, "p99": p99, "max_lat": max_lat,
        "param_mb": param_mb,
    }


def print_summary(model_name, model_run, stats):
    """Print summary for a single model."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"Model: {model_run['model_id']}")
    print(f"Device: {model_run['device']}")
    print(f"Parameters: {model_run['param_count']:,} ({model_run['param_mb']:.0f} MB at fp32)")
    print(f"Load time: {model_run['load_time_s']:.2f}s")
    print(f"Label mapping: {model_run['label_mapping']}")
    fc, fl = stats["filler_clean"], stats["filler_cases"]
    co, cl = stats["corr_clean"], stats["corr_cases"]
    mo, ml = stats["meaning_ok"], stats["total_cases"]
    po, pl = stats["pt_clean"], stats["pt_cases"]
    print(f"\nFiller removal rate:       {fc}/{fl} ({stats['filler_rate']:.0%})")
    print(f"Self-correction rate:      {co}/{cl} ({stats['corr_rate']:.0%})")
    print(f"Meaning preservation:      {mo}/{ml} ({stats['meaning_rate']:.0%})")
    print(f"Passthrough accuracy:      {po}/{pl} ({stats['pt_rate']:.0%})")
    print(f"Global FP rate:            {stats['global_fp_count']}/{stats['global_content_count']} ({stats['global_fp']:.1%})")
    print(f"\nLatency: p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  p99={stats['p99']:.1f}ms  max={stats['max_lat']:.1f}ms")


def print_criteria_scoring(model_name, stats):
    """Print hypothesis criteria pass/fail for a model."""
    fc = stats["filler_clean"]
    co = stats["corr_clean"]
    mo = stats["meaning_ok"]
    po = stats["pt_clean"]
    fl = stats["filler_cases"]
    cl = stats["corr_cases"]
    ml = stats["total_cases"]
    pl = stats["pt_cases"]
    global_fp = stats["global_fp"]
    p95 = stats["p95"]
    param_mb = stats["param_mb"]

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

    print(f"\n{'='*60}")
    print(f"HYPOTHESIS CRITERIA SCORING: {model_name}")
    print(f"{'='*60}")
    print(f"\n  --- Success Criteria ---")
    for desc, val, flag in [
        ("Filler detection >= 4/5", f"{fc}/{fl}", s1),
        ("Self-correction detection >= 1/2", f"{co}/{cl}", s2),
        ("Content preservation FP < 5%", f"{global_fp:.1%}", s3),
        ("Clean passthrough 2/2", f"{po}/{pl}", s4),
        ("Reassembly quality >= 8/12", f"{mo}/{ml}", s5),
        ("Latency < 50ms p95", f"{p95:.1f}ms", s6),
        ("Memory < 1GB", f"{param_mb:.0f}MB", s7),
    ]:
        st = "PASS" if flag else "FAIL"
        print(f"  [{st:>9}] {desc}: {val}")

    print(f"\n  --- Failure Criteria ---")
    for desc, val, flag in [
        ("Massive FP >= 10% content tokens", f"{global_fp:.1%}", f1),
        ("Filler blindness <= 2/5", f"{fc}/{fl}", f2),
        ("Complete self-correction failure 0/2", f"{co}/{cl}", f3),
        ("Unacceptable latency >= 200ms p95", f"{p95:.1f}ms", f4),
    ]:
        st = "TRIGGERED" if flag else "OK"
        print(f"  [{st:>9}] {desc}: {val}")

    return {
        "success": {
            "filler_detection": {"required": ">=4/5", "actual": f"{fc}/{fl}", "pass": s1},
            "self_correction": {"required": ">=1/2", "actual": f"{co}/{cl}", "pass": s2},
            "content_preservation": {"required": "FP<5%", "actual": f"{global_fp:.1%}", "pass": s3},
            "clean_passthrough": {"required": "2/2", "actual": f"{po}/{pl}", "pass": s4},
            "reassembly_quality": {"required": ">=8/12", "actual": f"{mo}/{ml}", "pass": s5},
            "latency": {"required": "<50ms p95", "actual": f"{p95:.1f}ms", "pass": s6},
            "memory": {"required": "<1GB", "actual": f"{param_mb:.0f}MB", "pass": s7},
        },
        "failure": {
            "massive_fp": {"threshold": ">=10%", "actual": f"{global_fp:.1%}", "triggered": f1},
            "filler_blindness": {"threshold": "<=2/5", "actual": f"{fc}/{fl}", "triggered": f2},
            "self_correction_failure": {"threshold": "0/2", "actual": f"{co}/{cl}", "triggered": f3},
            "unacceptable_latency": {"threshold": ">=200ms p95", "actual": f"{p95:.1f}ms", "triggered": f4},
        },
    }


def main() -> None:
    all_model_runs = {}
    all_stats = {}

    for model_name, model_id in MODELS:
        print(f"\n{'#' * 80}")
        print(f"# Loading and running: {model_name} ({model_id})")
        print(f"{'#' * 80}\n")

        model_run = run_model(model_name, model_id)
        all_model_runs[model_name] = model_run

        # Print per-model results and grids
        print_model_results(model_run)

        # Compute and print aggregate stats
        stats = compute_aggregate_stats(model_run)
        all_stats[model_name] = stats
        print_summary(model_name, model_run, stats)

        # Hypothesis criteria scoring
        criteria = print_criteria_scoring(model_name, stats)
        all_model_runs[model_name]["criteria"] = criteria

    # -- Side-by-side comparison table --
    print(f"\n{'=' * 100}")
    print(f"SIDE-BY-SIDE COMPARISON: base vs large vs Qwen 2B vs Qwen 4B")
    print(f"{'=' * 100}")

    base_s = all_stats.get("base", {})
    large_s = all_stats.get("large", {})

    def _fmt_ratio(ok, total):
        return f"{ok}/{total} ({ok/total:.0%})" if total else "N/A"

    print(f"\n{'Metric':<35} {'ModernBERT base':<18} {'ModernBERT large':<18} {'Qwen 2B':<15} {'Qwen 4B':<15}")
    print("-" * 101)
    comparisons = [
        ("Filler removal",
         _fmt_ratio(base_s.get("filler_clean", 0), base_s.get("filler_cases", 0)),
         _fmt_ratio(large_s.get("filler_clean", 0), large_s.get("filler_cases", 0)),
         "2/5 (40%)", "3/5 (60%)"),
        ("Self-correction",
         _fmt_ratio(base_s.get("corr_clean", 0), base_s.get("corr_cases", 0)),
         _fmt_ratio(large_s.get("corr_clean", 0), large_s.get("corr_cases", 0)),
         "0/2 (0%)", "0/2 (0%)"),
        ("Meaning preservation",
         _fmt_ratio(base_s.get("meaning_ok", 0), base_s.get("total_cases", 0)),
         _fmt_ratio(large_s.get("meaning_ok", 0), large_s.get("total_cases", 0)),
         "8/12 (67%)", "9/12 (75%)"),
        ("Passthrough",
         _fmt_ratio(base_s.get("pt_clean", 0), base_s.get("pt_cases", 0)),
         _fmt_ratio(large_s.get("pt_clean", 0), large_s.get("pt_cases", 0)),
         "2/2 (100%)", "2/2 (100%)"),
        ("Latency p95",
         f"{base_s.get('p95', 0):.0f}ms",
         f"{large_s.get('p95', 0):.0f}ms",
         "472ms", "~900ms"),
        ("Memory",
         f"{base_s.get('param_mb', 0):.0f}MB",
         f"{large_s.get('param_mb', 0):.0f}MB",
         "2,550 MB", "~5,000 MB"),
        ("Global FP rate",
         f"{base_s.get('global_fp', 0):.1%}",
         f"{large_s.get('global_fp', 0):.1%}",
         "N/A", "N/A"),
    ]
    for row in comparisons:
        metric, *vals = row
        print(f"  {metric:<35} {vals[0]:<18} {vals[1]:<18} {vals[2]:<15} {vals[3]:<15}")

    # -- Per-case comparison between base and large --
    print(f"\n{'=' * 120}")
    print(f"PER-CASE COMPARISON: base vs large")
    print(f"{'=' * 120}")
    hdr = (f"{'Name':<30} {'Category':<18} "
           f"{'base lat':>9} {'base sim':>9} {'base':>5} "
           f"{'large lat':>10} {'large sim':>10} {'large':>6}")
    print(hdr)
    print("-" * 120)
    base_results = all_model_runs.get("base", {}).get("results", [])
    large_results = all_model_runs.get("large", {}).get("results", [])
    for br, lr in zip(base_results, large_results):
        print(f"{br['name']:<30} {br['category']:<18} "
              f"{br['latency_ms']:>8.2f}ms {br['meaning_sim']:>8.3f} "
              f"{'PASS' if br['passed'] else 'FAIL':>5} "
              f"{lr['latency_ms']:>9.2f}ms {lr['meaning_sim']:>9.3f} "
              f"{'PASS' if lr['passed'] else 'FAIL':>6}")

    # -- Write JSON results --
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reviews_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    json_path = os.path.join(reviews_dir, f"run-{ts}.json")

    summary = {
        "timestamp": ts,
        "models": {},
        "prior_tests": [
            "qwen35-stt-cleanup (2B, disproven)",
            "qwen35-4b-cleanup (4B, disproven)",
        ],
        "num_test_cases": len(TEST_CASES),
    }

    for model_name in all_model_runs:
        mr = all_model_runs[model_name]
        st = all_stats[model_name]
        summary["models"][model_name] = {
            "model_id": mr["model_id"],
            "device": mr["device"],
            "param_count": mr["param_count"],
            "param_mb": mr["param_mb"],
            "load_time_s": mr["load_time_s"],
            "label_mapping": mr["label_mapping"],
            "aggregate": {
                "filler_removal_rate": round(st["filler_rate"], 3),
                "self_correction_rate": round(st["corr_rate"], 3),
                "meaning_preservation_rate": round(st["meaning_rate"], 3),
                "passthrough_accuracy": round(st["pt_rate"], 3),
                "global_fp_rate": round(st["global_fp"], 4),
                "global_fp_count": st["global_fp_count"],
                "global_content_count": st["global_content_count"],
            },
            "latency": {
                "p50_ms": round(st["p50"], 2),
                "p95_ms": round(st["p95"], 2),
                "p99_ms": round(st["p99"], 2),
                "max_ms": round(st["max_lat"], 2),
            },
            "hypothesis_criteria": mr.get("criteria", {}),
            "cases": [{k: r[k] for k in (
                "name", "category", "latency_ms", "total_tokens",
                "disfluent_count", "fluent_count", "disf_rate",
                "meaning_sim", "filler_clean", "passthrough_ok",
                "fp_rate", "fp_count", "content_count",
                "passed", "notes",
                "raw_input", "expected", "actual_output", "token_grid",
            )} for r in mr["results"]],
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to: {json_path}")


if __name__ == "__main__":
    main()
