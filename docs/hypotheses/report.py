"""Generate a markdown report comparing all hypothesis test results.

Usage:
    cd /Users/jud/Projects/operator
    .venv-poc/bin/python docs/hypotheses/report.py
    open docs/hypotheses/report.md
"""

import json
import os

BASE = "docs/hypotheses"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_first_json(directory):
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json"):
            return load_json(os.path.join(directory, fn))
    return None


def escape_md(text):
    """Escape pipe characters for markdown tables."""
    return text.replace("|", "\\|").replace("\n", " ")


def truncate(text, n=120):
    if len(text) <= n:
        return text
    return text[:n] + "…"


# --- Load all results ---

models = {}

# Qwen 2B
q2 = find_first_json(f"{BASE}/qwen35-stt-cleanup/working/reviews/")
if q2:
    models["Qwen3.5-2B"] = {
        "type": "Generative LLM",
        "params": "2B",
        "prompt": q2.get("system_prompt", ""),
        "cases": {
            c["name"]: {
                "raw": c["raw_input"],
                "expected": c["expected"],
                "output": c["actual_output"],
                "similarity": c.get("meaning_sim", 0),
                "latency_ms": c.get("latency_ms", 0),
                "passed": c.get("passed", False),
                "notes": c.get("notes", ""),
            }
            for c in q2["cases"]
        },
    }

# Qwen 4B
q4 = find_first_json(f"{BASE}/qwen35-4b-cleanup/working/reviews/")
if q4:
    models["Qwen3.5-4B"] = {
        "type": "Generative LLM",
        "params": "4B",
        "prompt": q4.get("system_prompt", ""),
        "cases": {
            c["name"]: {
                "raw": c["raw_input"],
                "expected": c["expected"],
                "output": c["actual_output"],
                "similarity": c.get("meaning_sim", 0),
                "latency_ms": c.get("latency_ms", 0),
                "passed": c.get("passed", False),
                "notes": c.get("notes", ""),
            }
            for c in q4["cases"]
        },
    }

# BERT (base + large)
bert = find_first_json(f"{BASE}/bert-disfluency/working/reviews/")
if bert and "models" in bert:
    for mkey, label in [("base", "ModernBERT-base"), ("large", "ModernBERT-large")]:
        if mkey in bert["models"]:
            mdata = bert["models"][mkey]
            models[label] = {
                "type": "Token classifier",
                "params": f"{mdata.get('param_count', 0) / 1e6:.0f}M",
                "prompt": "(token classification — no prompt)",
                "cases": {
                    c["name"]: {
                        "raw": c["raw_input"],
                        "expected": c["expected"],
                        "output": c["actual_output"],
                        "similarity": c.get("meaning_sim", 0),
                        "latency_ms": c.get("latency_ms", 0),
                        "passed": c.get("passed", False),
                        "notes": c.get("notes", ""),
                        "token_grid": c.get("token_grid", ""),
                    }
                    for c in mdata["cases"]
                },
            }

# CoEdit (best prompt per case)
coedit = load_json(f"{BASE}/coedit-cleanup/working/reviews/coedit-large-results.json")
if coedit:
    # For each case, find the prompt with highest similarity
    best_cases = {}
    for prompt_text, cases in coedit["prompts"].items():
        for c in cases:
            name = c["name"]
            sim = c.get("lev_sim", 0)
            if name not in best_cases or sim > best_cases[name]["similarity"]:
                best_cases[name] = {
                    "raw": c["raw"],
                    "expected": c["expected"],
                    "output": c["output"],
                    "similarity": sim,
                    "latency_ms": c.get("latency_ms", 0),
                    "passed": c.get("passed", False),
                    "notes": "",
                    "prompt_used": prompt_text,
                }
    # Also store all prompts for the per-prompt breakdown
    models["CoEdit-large"] = {
        "type": "Text editor (T5)",
        "params": "770M",
        "prompt": "Tested 4 prompts (see per-case details)",
        "cases": best_cases,
        "all_prompts": coedit["prompts"],
    }


# --- Generate report ---

lines = []
w = lines.append

w("# STT Cleanup Hypothesis Test Report")
w("")
w("Comparison of 5 models tested for cleaning up raw speech-to-text output.")
w("All models were evaluated on the same 12 test cases covering fillers,")
w("self-corrections, rambling speech, clean passthrough, short commands,")
w("and real WhisperKit fixture text.")
w("")

# Summary table
w("## Summary")
w("")
w("| Model | Type | Params | Filler Removal | Self-Correction | Meaning Preservation | Verdict |")
w("|---|---|---|---|---|---|---|")
for mname, mdata in models.items():
    cases = mdata["cases"]
    total = len(cases)
    passed = sum(1 for c in cases.values() if c["passed"])
    w(f"| {mname} | {mdata['type']} | {mdata['params']} | — | — | {passed}/{total} passed | **Disproven** |")
w("")

# Get the canonical case list from first model
case_names = list(next(iter(models.values()))["cases"].keys())

# Per-case comparison
w("## Per-Case Results")
w("")

for case_name in case_names:
    # Get raw/expected from first model that has it
    ref = None
    for mdata in models.values():
        if case_name in mdata["cases"]:
            ref = mdata["cases"][case_name]
            break
    if not ref:
        continue

    w(f"### {case_name}")
    w("")
    w(f"**Input:**")
    w(f"> {escape_md(ref['raw'])}")
    w("")
    w(f"**Expected:**")
    w(f"> {escape_md(ref['expected'])}")
    w("")

    w("| Model | Output | Similarity | Latency | Pass |")
    w("|---|---|---|---|---|")
    for mname, mdata in models.items():
        if case_name not in mdata["cases"]:
            continue
        c = mdata["cases"][case_name]
        output = escape_md(truncate(c["output"], 100))
        sim = f"{c['similarity']:.0%}" if isinstance(c['similarity'], float) else str(c['similarity'])
        lat = f"{c['latency_ms']:.0f}ms" if c['latency_ms'] else "—"
        icon = "pass" if c["passed"] else "**FAIL**"
        w(f"| {mname} | {output} | {sim} | {lat} | {icon} |")
    w("")

    # CoEdit prompt comparison for this case
    if "CoEdit-large" in models and "all_prompts" in models["CoEdit-large"]:
        w(f"**CoEdit prompt comparison:**")
        w("")
        w("| Prompt | Output | Similarity |")
        w("|---|---|---|")
        for prompt_text, cases in models["CoEdit-large"]["all_prompts"].items():
            for c in cases:
                if c["name"] == case_name:
                    output = escape_md(truncate(c["output"], 100))
                    w(f"| `{escape_md(prompt_text)}` | {output} | {c.get('lev_sim', 0):.0%} |")
        w("")

    # BERT token grid if available
    has_grid = False
    for mname in ["ModernBERT-base", "ModernBERT-large"]:
        if mname in models and case_name in models[mname]["cases"]:
            grid = models[mname]["cases"][case_name].get("token_grid", "")
            if grid:
                if not has_grid:
                    w("**BERT token labels:**")
                    w("")
                    has_grid = True
                w(f"_{mname}:_")
                w(f"```")
                w(grid[:500])
                w(f"```")
                w("")

# System prompts
w("## System Prompts / Instructions Used")
w("")
for mname, mdata in models.items():
    prompt = mdata.get("prompt", "")
    if prompt and prompt != "(token classification — no prompt)":
        w(f"### {mname}")
        w("")
        if mname == "CoEdit-large":
            w("CoEdit uses instruction prefixes prepended to input text:")
            w("")
            w("1. `Make this text fluent: <input>`")
            w("2. `Fix the grammar in this text: <input>`")
            w("3. `Remove disfluencies from this text: <input>`")
            w("4. `Fix the grammar and remove filler words: <input>`")
        else:
            w("```")
            w(prompt[:500])
            w("```")
        w("")

w("---")
w("*Generated from hypothesis test results in `docs/hypotheses/`*")

report = "\n".join(lines)
out_path = f"{BASE}/report.md"
with open(out_path, "w") as f:
    f.write(report)
print(f"Report written to {out_path} ({len(lines)} lines)")
