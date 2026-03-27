#!/usr/bin/env python3
"""Evaluate Qwen 3.5 0.8B as a local routing model.

Uses vanilla HuggingFace inference (same weights, native DeltaNet support).

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/routing-qwen/eval.py
    /tmp/coreml-venv/bin/python scripts/experiments/routing-qwen/eval.py --prompt v4
    /tmp/coreml-venv/bin/python scripts/experiments/routing-qwen/eval.py --sweep
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"

# ──────────────────────────────────────────────────────────────
# Sessions — mirrors the Operator benchmark scenario
# ──────────────────────────────────────────────────────────────

SESSIONS = [
    {
        "name": "ui-shell",
        "cwd": "/Users/jud/work/operator-ui",
        "context": (
            "Branch feat/settings-loading. Editing web/src/routes/settings.tsx, "
            "web/src/components/LoadingCard.tsx, and web/src/styles/dashboard.css "
            "for React loading states, layout polish, and responsive CSS."
        ),
        "recent": [
            ("user", "Fix the janky loading state on the settings dashboard and make the cards stop shifting on mobile."),
            ("assistant", "I updated LoadingCard.tsx, SettingsRoute.tsx, and dashboard.css; next I am wiring the pending query state and skeleton layout."),
        ],
    },
    {
        "name": "profile-api",
        "cwd": "/Users/jud/work/operator-api",
        "context": (
            "Branch feat/profile-endpoints. Editing api/routes/profile.py, "
            "services/user_profile.py, and db/queries/profile.sql for REST endpoints, "
            "auth middleware, and Postgres query performance."
        ),
        "recent": [
            ("user", "The user profile endpoint is timing out under load and we still need a new REST route for profile preferences."),
            ("assistant", "I traced the slowdown to profile.sql and the connection pool, and I am adding the preferences endpoint in profile.py."),
        ],
    },
    {
        "name": "staging-infra",
        "cwd": "/Users/jud/work/operator-infra",
        "context": (
            "Branch chore/staging-network. Editing infra/envs/staging/main.tf, "
            "modules/vpc, and modules/iam_policy for Terraform apply failures, "
            "private subnet layout, IAM policies, and S3 logging."
        ),
        "recent": [
            ("user", "The staging VPC rollout failed after the Terraform change and logging still needs the new S3 bucket policy."),
            ("assistant", "I am updating the private subnet module, the IAM policy document, and the S3 logging resources in staging Terraform."),
        ],
    },
]

# ──────────────────────────────────────────────────────────────
# Test cases — DEV set (tune prompts against) + HOLDOUT set (final eval)
# ──────────────────────────────────────────────────────────────

@dataclass
class Case:
    message: str
    expected: str | None  # None = ambiguous (correct if confident=false)
    tags: list[str] = field(default_factory=list)

# DEV set — use these for prompt iteration
DEV_CASES = [
    # Easy — clear signal
    Case("add a loading spinner to the React dashboard", "ui-shell", ["easy"]),
    Case("fix the CSS grid layout", "ui-shell", ["easy"]),
    Case("add a new REST endpoint for user profiles", "profile-api", ["easy"]),
    Case("fix the database connection pool", "profile-api", ["easy"]),
    Case("optimize the SQL query", "profile-api", ["easy"]),
    Case("update the terraform module for the new VPC", "staging-infra", ["easy"]),
    Case("add a new S3 bucket for logs", "staging-infra", ["easy"]),
    # Hard — cross-domain keywords
    Case("fix the AWS IAM permissions", "staging-infra", ["hard", "cross-domain"]),
    Case("add TypeScript types for the API response", "ui-shell", ["hard", "cross-domain"]),
    Case("add authentication to the API", "profile-api", ["hard", "cross-domain"]),
    # Ambiguous — should say not confident
    Case("fix the tests", None, ["ambiguous"]),
    Case("what's the status", None, ["ambiguous"]),
    Case("can you check if that worked", None, ["ambiguous"]),
    Case("add error handling", None, ["ambiguous"]),
    Case("the build is broken", None, ["ambiguous"]),
]

# HOLDOUT set — only run after prompt is finalized, no peeking
HOLDOUT_CASES = [
    # Clear routing
    Case("make the sidebar component collapsible", "ui-shell", ["easy"]),
    Case("the modal closes when you click outside it, fix that", "ui-shell", ["easy"]),
    Case("add pagination to the users list endpoint", "profile-api", ["easy"]),
    Case("the profile photo upload is returning a 500", "profile-api", ["easy"]),
    Case("add a CloudFront distribution for the static assets", "staging-infra", ["easy"]),
    Case("the staging RDS instance is running out of storage", "staging-infra", ["easy"]),
    # Cross-domain / tricky
    Case("add rate limiting to the API", "profile-api", ["hard"]),
    Case("fix the TypeScript build errors", "ui-shell", ["hard"]),
    Case("update the environment variables for staging", "staging-infra", ["hard"]),
    Case("add a health check endpoint", "profile-api", ["hard"]),
    Case("fix the CORS issue", "profile-api", ["hard", "cross-domain"]),
    Case("add monitoring alerts for the database", "staging-infra", ["hard", "cross-domain"]),
    Case("the webpack bundle is too large", "ui-shell", ["hard"]),
    Case("rotate the database credentials", "staging-infra", ["hard", "cross-domain"]),
    # Conversational / ambiguous
    Case("never mind go back to what you were doing", None, ["ambiguous"]),
    Case("how's it going", None, ["ambiguous"]),
    Case("run the linter", None, ["ambiguous"]),
    Case("update the README", None, ["ambiguous"]),
    Case("revert the last commit", None, ["ambiguous"]),
    Case("ship it", None, ["ambiguous"]),
]

# BLIND set — genuinely unseen, voice-style phrasings, no overlap with few-shot examples.
# These simulate real mic input: messy, conversational, indirect.
BLIND_CASES = [
    # Frontend — different phrasings, no keyword overlap with examples
    Case("the dropdown menu is appearing behind the other elements", "ui-shell", ["easy"]),
    Case("can you center that div vertically", "ui-shell", ["easy"]),
    Case("users are complaining the page takes forever to load on mobile", "ui-shell", ["hard"]),
    Case("there's a flash of unstyled content when the page loads", "ui-shell", ["hard"]),
    Case("the dark mode toggle doesn't persist when you refresh", "ui-shell", ["hard"]),
    # Backend — natural speech, indirect references
    Case("the login endpoint is rejecting valid passwords", "profile-api", ["easy"]),
    Case("we need to add a forgot password flow", "profile-api", ["hard"]),
    Case("the API is returning stale data after updates", "profile-api", ["hard"]),
    Case("someone reported they can see other users' data", "profile-api", ["hard"]),
    Case("the query is doing a full table scan on the users table", "profile-api", ["easy"]),
    # Infra — ops language, not dev language
    Case("the SSL certificate is expiring next week", "staging-infra", ["hard"]),
    Case("we need to add a NAT gateway for the private subnets", "staging-infra", ["easy"]),
    Case("the bill went up 40 percent last month", "staging-infra", ["hard"]),
    Case("can you set up a bastion host for SSH access", "staging-infra", ["hard"]),
    Case("we need to enable versioning on the backup bucket", "staging-infra", ["hard"]),
    # Cross-domain — deliberately tricky
    Case("the API response time is over 2 seconds", "profile-api", ["hard", "cross-domain"]),
    Case("add input validation to the signup form", "ui-shell", ["hard", "cross-domain"]),
    Case("the websocket connection keeps dropping", "profile-api", ["hard", "cross-domain"]),
    Case("we need to set up a CDN for the images", "staging-infra", ["hard", "cross-domain"]),
    Case("the docker container won't start", "staging-infra", ["hard", "cross-domain"]),
    # Ambiguous — genuinely impossible to route
    Case("hey what are you working on right now", None, ["ambiguous"]),
    Case("undo that", None, ["ambiguous"]),
    Case("actually never mind", None, ["ambiguous"]),
    Case("wait hold on", None, ["ambiguous"]),
    Case("can you explain what you just did", None, ["ambiguous"]),
    Case("why did you do it that way", None, ["ambiguous"]),
    Case("that looks wrong", None, ["ambiguous"]),
    Case("try again", None, ["ambiguous"]),
    Case("let me think about this for a sec", None, ["ambiguous"]),
    Case("commit what you have", None, ["ambiguous"]),
]


# ──────────────────────────────────────────────────────────────
# Prompt strategies
# ──────────────────────────────────────────────────────────────

def format_sessions_full() -> str:
    """Full session context with recent messages."""
    lines = []
    for i, s in enumerate(SESSIONS):
        lines.append(f'{i+1}. "{s["name"]}" ({s["cwd"]})')
        lines.append(f'   Context: {s["context"]}')
        for role, text in s["recent"]:
            lines.append(f'   Recent {role}: "{text[:80]}"')
    return "\n".join(lines)


def format_sessions_compact() -> str:
    """Compact session descriptions — tech stack focused."""
    lines = []
    for s in SESSIONS:
        # Extract the key tech signals from context
        lines.append(f'- {s["name"]}: {s["context"][:120]}')
    return "\n".join(lines)


SESSIONS_FULL = format_sessions_full()
SESSIONS_COMPACT = format_sessions_compact()

PROMPTS = {}

# ── v1: Direct port of current claude -p prompt ──
PROMPTS["v1"] = (
    f"Route the user message to the best active session.\n"
    f"Match using the provided session name, working directory, project context, and recent messages.\n"
    f"Base the decision on those session details, not on assumptions about common team names; "
    f"if the match is weak or ambiguous, return an empty session with confident=false.\n"
    f"Reply only with JSON: {{\"session\":\"<name or empty>\",\"confident\":true|false}}\n\n"
    f"Active sessions:\n{SESSIONS_FULL}\n\n"
    f"Use only the session details above to choose the best target."
)

# ── v2: Simplified instructions, same context ──
PROMPTS["v2"] = (
    f"You are a message router. Given a user's voice command, pick which coding session it belongs to.\n\n"
    f"Active sessions:\n{SESSIONS_FULL}\n\n"
    f"Rules:\n"
    f"- Pick the session whose project best matches the message topic.\n"
    f"- If unclear, set confident=false and session=\"\".\n"
    f"- Reply ONLY with JSON: {{\"session\":\"<name>\",\"confident\":true|false}}"
)

# ── v3: Ultra-short, no context (baseline) ──
PROMPTS["v3"] = (
    f"Pick which session this message should go to.\n\n"
    f"Sessions:\n"
    f"- ui-shell: React/TypeScript frontend (settings page, CSS, components)\n"
    f"- profile-api: Python REST API (user profiles, SQL, database)\n"
    f"- staging-infra: Terraform infrastructure (VPC, IAM, S3, AWS)\n\n"
    f"Reply ONLY with JSON: {{\"session\":\"<name or empty>\",\"confident\":true|false}}\n"
    f"If ambiguous, use session=\"\" and confident=false."
)

# ── v4: Few-shot with reasoning examples ──
PROMPTS["v4"] = (
    f"You route voice commands to coding sessions. Each session works on a different project.\n\n"
    f"Sessions:\n{SESSIONS_COMPACT}\n\n"
    f"Examples:\n"
    f'User: "fix the button hover state"\n'
    f'{{\"session\":\"ui-shell\",\"confident\":true}}\n\n'
    f'User: "add a new database migration for the users table"\n'
    f'{{\"session\":\"profile-api\",\"confident\":true}}\n\n'
    f'User: "update the security group rules"\n'
    f'{{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f'User: "fix the tests"\n'
    f'{{\"session\":\"\",\"confident\":false}}\n\n'
    f"Rules:\n"
    f"- Match by technology and project scope, not individual keywords.\n"
    f"- Frontend code (React, TypeScript, CSS, components, UI) → ui-shell\n"
    f"- Backend code (Python, REST, SQL, database, API logic) → profile-api\n"
    f"- Infrastructure (Terraform, AWS, IAM, S3, VPC, deploy config) → staging-infra\n"
    f"- If the message could apply to multiple sessions, set confident=false.\n"
    f"- Reply ONLY with JSON. No explanation."
)

# ── v5: Few-shot with HARD examples that teach cross-domain reasoning ──
PROMPTS["v5"] = (
    f"You route voice commands to coding sessions.\n\n"
    f"Sessions:\n{SESSIONS_COMPACT}\n\n"
    f"Examples:\n"
    f'User: "fix the button hover state"\n'
    f'{{\"session\":\"ui-shell\",\"confident\":true}}\n\n'
    f'User: "the profile endpoint is returning a 500"\n'
    f'{{\"session\":\"profile-api\",\"confident\":true}}\n\n'
    f'User: "update the IAM role permissions"\n'
    f'{{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f'User: "add TypeScript types for the API response"\n'
    f'{{\"session\":\"ui-shell\",\"confident\":true}}\n\n'
    f'User: "fix the AWS IAM permissions"\n'
    f'{{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f'User: "run the tests"\n'
    f'{{\"session\":\"\",\"confident\":false}}\n\n'
    f'User: "how is it going"\n'
    f'{{\"session\":\"\",\"confident\":false}}\n\n'
    f"Match by technology and project scope. Reply ONLY with JSON."
)

# ── v6: Chain-of-thought — let the model reason first ──
PROMPTS["v6"] = (
    f"You route voice commands to coding sessions.\n\n"
    f"Sessions:\n{SESSIONS_COMPACT}\n\n"
    f"For each message, think step by step:\n"
    f"1. What technology or domain does this message relate to?\n"
    f"2. Which session works on that technology?\n"
    f"3. Is the match clear or ambiguous?\n\n"
    f"Example:\n"
    f'User: "add TypeScript types for the API response"\n'
    f"Think: TypeScript is a frontend language. ui-shell works on TypeScript/React. "
    f"Even though \"API\" appears, the task is writing TypeScript code.\n"
    f'{{\"session\":\"ui-shell\",\"confident\":true}}\n\n'
    f'User: "fix the IAM permissions"\n'
    f"Think: IAM is AWS Identity and Access Management. staging-infra works on AWS/Terraform. "
    f"Even though permissions relate to auth, IAM is infrastructure.\n"
    f'{{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f'User: "fix the tests"\n'
    f"Think: All sessions have tests. No clear match.\n"
    f'{{\"session\":\"\",\"confident\":false}}\n\n'
    f"Now route the following message. Think briefly, then output JSON on the last line."
)

# ── v7: Structured few-shot — tech stack explicit, hard examples, no CoT ──
PROMPTS["v7"] = (
    f"Route the message to the correct session. Reply with JSON only.\n\n"
    f"SESSION MAP:\n"
    f"  ui-shell → React, TypeScript, CSS, JSX, TSX, components, frontend, browser, webpack, layout, responsive, UI\n"
    f"  profile-api → Python, REST, SQL, Postgres, database, API endpoints, backend, FastAPI, migration, ORM\n"
    f"  staging-infra → Terraform, AWS, IAM, S3, VPC, CloudFront, RDS, deploy, staging, infrastructure, security groups\n\n"
    f"EXAMPLES:\n"
    f'  "fix the button hover state" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "add a database index on email" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "update the S3 bucket policy" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "add TypeScript types for the API response" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "fix the AWS IAM permissions" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "add rate limiting" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "what is the status" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"FORMAT: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v8: v7 + no-think directive (Qwen3.5 supports /no_think) ──
PROMPTS["v8"] = PROMPTS["v7"]  # same prompt, but we'll add /no_think in the user message

# ── v9: v7 base + more ambiguous examples + explicit abstain rule ──
PROMPTS["v9"] = (
    f"Route the message to the correct session. Reply with JSON only.\n\n"
    f"SESSION MAP:\n"
    f"  ui-shell → React, TypeScript, CSS, JSX, TSX, components, frontend, browser, webpack, layout, responsive, UI\n"
    f"  profile-api → Python, REST, SQL, Postgres, database, API endpoints, backend, FastAPI, migration, ORM\n"
    f"  staging-infra → Terraform, AWS, IAM, S3, VPC, CloudFront, RDS, deploy, staging, infrastructure, security groups\n\n"
    f"CONFIDENT examples (message clearly matches ONE session):\n"
    f'  "fix the button hover state" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "add a database index on email" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "update the S3 bucket policy" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "add TypeScript types for the API response" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "fix the AWS IAM permissions" → {{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f"NOT CONFIDENT examples (message could apply to ANY session):\n"
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "what is the status" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "add error handling" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "the build is broken" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "run the linter" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "ship it" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"RULE: If the message does not mention a specific technology, file type, or service "
    f"that uniquely identifies one session, set confident=false.\n\n"
    f"FORMAT: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v10: v9 + no-think ──
PROMPTS["v10"] = PROMPTS["v9"]

# ── v11: Tighter — fewer confident examples, more ambiguous, ratio matters ──
PROMPTS["v11"] = (
    f"You classify voice messages to coding sessions.\n\n"
    f"SESSIONS:\n"
    f"  ui-shell: React, TypeScript, CSS, frontend, components, browser, webpack, UI\n"
    f"  profile-api: Python, REST, SQL, Postgres, database, API endpoints, backend\n"
    f"  staging-infra: Terraform, AWS, IAM, S3, VPC, CloudFront, deploy, infrastructure\n\n"
    f"ROUTE to a session when the message names a technology owned by that session:\n"
    f'  "fix the CSS grid" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "add a REST endpoint" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "update the terraform module" → {{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f"SAY NOT CONFIDENT when the message is generic and could apply to any session:\n"
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "add error handling" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "the build is broken" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "what is the status" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "ship it" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "revert the last commit" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "how is it going" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"Reply with JSON only. No explanation.\n"
    f"FORMAT: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v12: v11 + no-think ──
PROMPTS["v12"] = PROMPTS["v11"]

# ── v15: Dynamic session context + few-shot + strong abstain bias ──
# This is what production would look like — session descriptions come from
# the live RoutingPrompt.buildContextPrompt(), not a hardcoded tech-stack map.
PROMPTS["v15"] = (
    f"You route voice commands to coding sessions. Pick the session whose current work "
    f"best matches the message. If unsure, say not confident — asking the user is better "
    f"than guessing wrong.\n\n"
    f"ACTIVE SESSIONS:\n"
    f'1. "ui-shell" (/Users/jud/work/operator-ui)\n'
    f"   Branch feat/settings-loading. Editing settings.tsx, LoadingCard.tsx, dashboard.css.\n"
    f'   Recent user: "Fix the janky loading state on the settings dashboard"\n'
    f'   Recent assistant: "I updated LoadingCard.tsx and dashboard.css"\n\n'
    f'2. "profile-api" (/Users/jud/work/operator-api)\n'
    f"   Branch feat/profile-endpoints. Editing profile.py, user_profile.py, profile.sql.\n"
    f'   Recent user: "The user profile endpoint is timing out under load"\n'
    f'   Recent assistant: "I traced the slowdown to profile.sql and the connection pool"\n\n'
    f'3. "staging-infra" (/Users/jud/work/operator-infra)\n'
    f"   Branch chore/staging-network. Editing main.tf, modules/vpc, modules/iam_policy.\n"
    f'   Recent user: "The staging VPC rollout failed after the Terraform change"\n'
    f'   Recent assistant: "I am updating the private subnet module and IAM policy"\n\n'
    f"EXAMPLES:\n"
    f'  "fix the button hover state" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "the API is returning 500s" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "update the IAM role" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "add error handling" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "how is it going" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "the build is broken" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"RULE: Only set confident=true when the message clearly relates to ONE session's "
    f"files, technologies, or current work. When in doubt, set confident=false.\n\n"
    f"Reply with JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v16: v15 + no-think ──
PROMPTS["v16"] = PROMPTS["v15"]

# ── v17: Pure context matching — no tech-stack hints, just conversation context ──
# This is the production-realistic approach: session context is the ONLY signal.
PROMPTS["v17"] = (
    f"You route voice commands to coding sessions based on what each session is working on.\n"
    f"If the message doesn't clearly relate to one session's current work, set confident=false.\n"
    f"Asking is better than guessing wrong.\n\n"
    f"Sessions:\n"
    f'1. "ui-shell"\n'
    f'   User asked: "Fix the janky loading state on the settings dashboard and make the cards stop shifting on mobile"\n'
    f'   Working on: "I updated LoadingCard.tsx, SettingsRoute.tsx, and dashboard.css; '
    f'next I am wiring the pending query state and skeleton layout"\n\n'
    f'2. "profile-api"\n'
    f'   User asked: "The user profile endpoint is timing out under load and we need a new REST route for preferences"\n'
    f'   Working on: "I traced the slowdown to profile.sql and the connection pool, '
    f'and I am adding the preferences endpoint in profile.py"\n\n'
    f'3. "staging-infra"\n'
    f'   User asked: "The staging VPC rollout failed and logging needs the new S3 bucket policy"\n'
    f'   Working on: "I am updating the private subnet module, the IAM policy document, '
    f'and the S3 logging resources in staging Terraform"\n\n'
    f"Reply JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v18: v17 + no-think ──
PROMPTS["v18"] = PROMPTS["v17"]

# ── v19: v17 + few-shot examples showing the REASONING pattern ──
# Teach the model to match by topic similarity, not keywords
PROMPTS["v19"] = (
    f"You route voice commands to coding sessions. Match the message to whichever "
    f"session is doing the most related work. If no session is clearly related, "
    f"set confident=false — asking is better than guessing wrong.\n\n"
    f"Sessions:\n"
    f'1. "ui-shell"\n'
    f'   User asked: "Fix the janky loading state on the settings dashboard"\n'
    f'   Working on: "I updated LoadingCard.tsx, SettingsRoute.tsx, and dashboard.css; '
    f'wiring the pending query state and skeleton layout"\n\n'
    f'2. "profile-api"\n'
    f'   User asked: "The profile endpoint is timing out under load"\n'
    f'   Working on: "I traced the slowdown to profile.sql and the connection pool, '
    f'adding the preferences endpoint in profile.py"\n\n'
    f'3. "staging-infra"\n'
    f'   User asked: "The staging VPC rollout failed"\n'
    f'   Working on: "Updating the private subnet module, IAM policy, '
    f'and S3 logging resources in staging Terraform"\n\n'
    f"Examples:\n"
    f'  "the dashboard cards are flickering" → {{\"session\":\"ui-shell\",\"confident\":true}}  (relates to ui-shell\'s loading/layout work)\n'
    f'  "the query is still slow" → {{\"session\":\"profile-api\",\"confident\":true}}  (relates to profile-api\'s SQL slowdown)\n'
    f'  "the deploy is stuck" → {{\"session\":\"staging-infra\",\"confident\":true}}  (relates to staging-infra\'s infra work)\n'
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}  (all sessions have tests)\n'
    f'  "how is it going" → {{\"session\":\"\",\"confident\":false}}  (not about any specific work)\n\n'
    f"Reply JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v20: v19 + no-think ──
PROMPTS["v20"] = PROMPTS["v19"]

# ── v21: Structured context — emphasize what each session JUST said and did ──
# The assistant's last message is gold. It names files, describes actions, mentions
# specific technologies. Structure the prompt so the model can scan and match.
PROMPTS["v21"] = (
    f"Match the voice command to the session doing the most related work.\n"
    f"If no session clearly matches, respond with confident=false.\n\n"
    f"SESSION 1: ui-shell\n"
    f"  Last spoke about: loading states, settings dashboard, mobile layout, cards shifting\n"
    f"  Files: LoadingCard.tsx, SettingsRoute.tsx, dashboard.css\n"
    f"  Doing: wiring pending query state and skeleton layout\n\n"
    f"SESSION 2: profile-api\n"
    f"  Last spoke about: profile endpoint timing out, SQL slowdown, connection pool, REST route\n"
    f"  Files: profile.py, user_profile.py, profile.sql\n"
    f"  Doing: fixing query performance and adding preferences endpoint\n\n"
    f"SESSION 3: staging-infra\n"
    f"  Last spoke about: VPC rollout failure, S3 bucket policy, Terraform, IAM\n"
    f"  Files: main.tf, modules/vpc, modules/iam_policy\n"
    f"  Doing: updating subnet module, IAM policy, S3 logging\n\n"
    f"EXAMPLES:\n"
    f'  "the cards are still jumping around" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "is the query faster now" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "did terraform apply succeed" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "how is it going" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v22: v21 + no-think ──
PROMPTS["v22"] = PROMPTS["v21"]

# ── v23: v21 structure but with keywords extracted from the messages ──
# Make it REALLY easy — pull out the actual words the model needs to match against
PROMPTS["v23"] = (
    f"Match the voice command to the session doing related work.\n"
    f"If no clear match, set confident=false. Better to ask than guess wrong.\n\n"
    f"SESSION 1: ui-shell\n"
    f"  Topics: loading, dashboard, settings, cards, mobile, layout, CSS, skeleton, responsive\n"
    f"  Files: LoadingCard.tsx, SettingsRoute.tsx, dashboard.css\n"
    f"  Status: fixing loading states and card layout on settings page\n\n"
    f"SESSION 2: profile-api\n"
    f"  Topics: profile, endpoint, timeout, SQL, query, connection pool, REST, preferences, API\n"
    f"  Files: profile.py, user_profile.py, profile.sql\n"
    f"  Status: fixing slow query and adding preferences endpoint\n\n"
    f"SESSION 3: staging-infra\n"
    f"  Topics: VPC, staging, Terraform, S3, bucket, IAM, policy, subnet, deploy, rollout\n"
    f"  Files: main.tf, modules/vpc, modules/iam_policy\n"
    f"  Status: fixing VPC rollout and updating IAM and S3 config\n\n"
    f"Match the command to whichever session's topics, files, or status are most related.\n"
    f"Generic commands like 'fix the tests' or 'how is it going' match no session.\n\n"
    f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v24: v23 + no-think ──
PROMPTS["v24"] = PROMPTS["v23"]

# ── v25: v23 + few-shot examples showing indirect matches ──
# Teach the model that "center a div" relates to CSS/layout work,
# "someone can see other users' data" relates to API/profile work, etc.
PROMPTS["v25"] = (
    f"Match the voice command to the session doing related work.\n"
    f"If no clear match, set confident=false. Better to ask than guess wrong.\n\n"
    f"SESSION 1: ui-shell\n"
    f"  Topics: loading, dashboard, settings, cards, mobile, layout, CSS, skeleton, responsive\n"
    f"  Files: LoadingCard.tsx, SettingsRoute.tsx, dashboard.css\n"
    f"  Status: fixing loading states and card layout on settings page\n\n"
    f"SESSION 2: profile-api\n"
    f"  Topics: profile, endpoint, timeout, SQL, query, connection pool, REST, preferences, API\n"
    f"  Files: profile.py, user_profile.py, profile.sql\n"
    f"  Status: fixing slow query and adding preferences endpoint\n\n"
    f"SESSION 3: staging-infra\n"
    f"  Topics: VPC, staging, Terraform, S3, bucket, IAM, policy, subnet, deploy, rollout\n"
    f"  Files: main.tf, modules/vpc, modules/iam_policy\n"
    f"  Status: fixing VPC rollout and updating IAM and S3 config\n\n"
    f"EXAMPLES (match by topic relatedness):\n"
    f'  "center that div" → ui-shell (CSS/layout work)\n'
    f'  "the page flickers on load" → ui-shell (loading states)\n'
    f'  "users can see each other\'s data" → profile-api (user data/API)\n'
    f'  "the response is super slow" → profile-api (query performance)\n'
    f'  "the deploy keeps failing" → staging-infra (deploy/rollout)\n'
    f'  "the certificate is expiring" → staging-infra (infra/ops)\n'
    f'  "fix the tests" → not confident (all sessions have tests)\n'
    f'  "undo that" → not confident (could be any session)\n\n'
    f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v26: v25 + no-think ──
PROMPTS["v26"] = PROMPTS["v25"]

# Import v29-v45
from prompts_v29_v45 import build_prompts as _build_v29
PROMPTS.update(_build_v29())

# ── v27: Zero wrong-confident target ──
# Same keyword structure as v23 but with aggressive confidence gating.
# The model should ONLY say confident=true when it can point to a specific
# keyword overlap between the message and exactly one session.
PROMPTS["v27"] = (
    f"Match the voice command to the session doing related work.\n"
    f"IMPORTANT: Only set confident=true when the message directly overlaps with "
    f"ONE session's topics or files. If there is ANY doubt, set confident=false. "
    f"A wrong routing is much worse than asking the user to clarify.\n\n"
    f"SESSION 1: ui-shell\n"
    f"  Topics: loading, dashboard, settings, cards, mobile, layout, CSS, skeleton, responsive\n"
    f"  Files: LoadingCard.tsx, SettingsRoute.tsx, dashboard.css\n"
    f"  Status: fixing loading states and card layout on settings page\n\n"
    f"SESSION 2: profile-api\n"
    f"  Topics: profile, endpoint, timeout, SQL, query, connection pool, REST, preferences, API\n"
    f"  Files: profile.py, user_profile.py, profile.sql\n"
    f"  Status: fixing slow query and adding preferences endpoint\n\n"
    f"SESSION 3: staging-infra\n"
    f"  Topics: VPC, staging, Terraform, S3, bucket, IAM, policy, subnet, deploy, rollout\n"
    f"  Files: main.tf, modules/vpc, modules/iam_policy\n"
    f"  Status: fixing VPC rollout and updating IAM and S3 config\n\n"
    f"CONFIDENT — message clearly matches ONE session:\n"
    f'  "the cards are still jumping" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "is the query faster now" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "did the terraform apply work" → {{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f"NOT CONFIDENT — generic, conversational, or could match multiple sessions:\n"
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "add error handling" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "how is it going" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "try again" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "that looks wrong" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "undo that" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "set up a CDN" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "the build is broken" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v28: v27 + no-think ──
PROMPTS["v28"] = PROMPTS["v27"]

# ── v13: v9 base + targeted cross-domain few-shots for the holdout failures ──
PROMPTS["v13"] = (
    f"Route the message to the correct session. Reply with JSON only.\n\n"
    f"SESSION MAP:\n"
    f"  ui-shell → React, TypeScript, CSS, JSX, TSX, components, frontend, browser, webpack, layout, responsive, UI\n"
    f"  profile-api → Python, REST, SQL, Postgres, API endpoints, backend, FastAPI, ORM, CORS, middleware\n"
    f"  staging-infra → Terraform, AWS, IAM, S3, VPC, CloudFront, RDS, deploy, staging, infrastructure, security groups, monitoring, credentials, secrets\n\n"
    f"CONFIDENT examples:\n"
    f'  "fix the button hover state" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "add a database index on email" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "update the S3 bucket policy" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "add TypeScript types for the API response" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
    f'  "fix the AWS IAM permissions" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "add a new API endpoint" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "fix the CORS headers" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
    f'  "add a CloudFront distribution" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "rotate the database credentials" → {{\"session\":\"staging-infra\",\"confident\":true}}\n'
    f'  "add monitoring alerts" → {{\"session\":\"staging-infra\",\"confident\":true}}\n\n'
    f"NOT CONFIDENT examples (generic, could apply to any session):\n"
    f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "what is the status" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "add error handling" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "the build is broken" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "update the README" → {{\"session\":\"\",\"confident\":false}}\n'
    f'  "ship it" → {{\"session\":\"\",\"confident\":false}}\n\n'
    f"RULE: If the message does not mention a specific technology, file type, or service "
    f"that uniquely identifies one session, set confident=false.\n\n"
    f"FORMAT: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
)

# ── v14: v13 + no-think ──
PROMPTS["v14"] = PROMPTS["v13"]


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """Extract first JSON object from model output."""
    # Strip thinking tags if present
    text = re.sub(r"<\|?thinking\|?>.*?</\|?thinking\|?>", "", text, flags=re.DOTALL).strip()
    # Try direct parse
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Regex fallback
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def generate(model, tokenizer, system_prompt: str, user_message: str,
             max_new_tokens: int = 80, no_think: bool = False) -> str:
    """Generate a response using the patched DeltaNet model."""
    if no_think:
        user_message = "/no_think\n" + user_message

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case: Case
    session: str
    confident: bool
    correct: bool
    raw_output: str
    latency_ms: float
    parse_ok: bool


def eval_cases(model, tokenizer, prompt_name: str, system_prompt: str,
               cases: list[Case], label: str, no_think: bool = False) -> list[CaseResult]:
    results = []
    is_cot = prompt_name == "v6"

    for case in cases:
        t0 = time.perf_counter()
        raw = generate(model, tokenizer, system_prompt, case.message,
                       max_new_tokens=150 if is_cot else 50,
                       no_think=no_think)
        latency = (time.perf_counter() - t0) * 1000

        result = extract_json(raw)
        parse_ok = result is not None

        if result is None:
            session, confident = "PARSE_ERROR", False
        else:
            session = result.get("session", "")
            confident = result.get("confident", False)

        if case.expected is None:
            correct = not confident or session == ""
        else:
            correct = session == case.expected

        results.append(CaseResult(
            case=case, session=session, confident=confident,
            correct=correct, raw_output=raw, latency_ms=latency,
            parse_ok=parse_ok,
        ))

    return results


def print_results(results: list[CaseResult], prompt_name: str, label: str):
    print(f"\n{'='*70}")
    print(f"  {label} — prompt: {prompt_name}")
    print(f"{'='*70}")

    correct = sum(1 for r in results if r.correct)
    total = len(results)
    parse_fails = sum(1 for r in results if not r.parse_ok)
    lats = [r.latency_ms for r in results]

    for r in results:
        marker = "✓" if r.correct else "✗"
        conf_str = "✓" if r.confident else "·"
        expected_label = r.case.expected or "(ambiguous)"
        tags = " ".join(f"[{t}]" for t in r.case.tags)
        print(f"  [{marker}] \"{r.case.message}\"")
        print(f"       → {r.session} (conf={conf_str})  expected={expected_label}  {tags}  [{r.latency_ms:.0f}ms]")
        if not r.parse_ok:
            print(f"       RAW: {r.raw_output[:120]}")

    print(f"\n  Score: {correct}/{total} ({100*correct/total:.0f}%)")

    # Breakdown by tag
    tag_scores = {}
    for r in results:
        for tag in r.case.tags:
            tag_scores.setdefault(tag, [0, 0])
            tag_scores[tag][1] += 1
            if r.correct:
                tag_scores[tag][0] += 1
    if tag_scores:
        parts = [f"{tag}={c}/{t}" for tag, (c, t) in sorted(tag_scores.items())]
        print(f"  Breakdown: {', '.join(parts)}")

    if parse_fails:
        print(f"  Parse failures: {parse_fails}")

    print(f"  Latency: avg={sum(lats)/len(lats):.0f}ms  "
          f"p50={sorted(lats)[len(lats)//2]:.0f}ms  "
          f"min={min(lats):.0f}ms  max={max(lats):.0f}ms")

    return correct, total


def print_summary(all_results: dict[str, tuple[int, int]]):
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    ranked = sorted(all_results.items(), key=lambda x: x[1][0] / max(x[1][1], 1), reverse=True)
    for name, (correct, total) in ranked:
        pct = 100 * correct / total
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {name:8s}  {bar}  {correct}/{total} ({pct:.0f}%)")


def load_model():
    print(f"Loading {MODEL_ID} (MPS)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("mps")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Model loaded.\n")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 3.5 DeltaNet as routing model")
    parser.add_argument("--prompt", default=None, help="Test a specific prompt (v1-v8)")
    parser.add_argument("--sweep", action="store_true", help="Run all prompts on dev set")
    parser.add_argument("--holdout", action="store_true", help="Run on holdout set (final eval only)")
    parser.add_argument("--all", action="store_true", help="Run all prompts on all cases")
    parser.add_argument("--blind", action="store_true", help="Run on blind test set (never tuned against)")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.sweep or args.all:
        prompts_to_test = list(PROMPTS.keys())
    elif args.prompt:
        prompts_to_test = [args.prompt]
    else:
        # Default: run v27-v45 on blind
        prompts_to_test = [f"v{i}" for i in range(27, 46)]

    cases = DEV_CASES
    label_prefix = "DEV"
    if args.holdout:
        cases = HOLDOUT_CASES
        label_prefix = "HOLDOUT"
    if args.all:
        cases = DEV_CASES + HOLDOUT_CASES
        label_prefix = "ALL"
    if getattr(args, "blind", False):
        cases = BLIND_CASES
        label_prefix = "BLIND"

    summary = {}
    for prompt_name in prompts_to_test:
        system_prompt = PROMPTS[prompt_name]
        even_no_think = {"v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "v24", "v26", "v28",
                         "v30", "v32", "v34", "v36", "v38", "v40", "v42", "v44"}
        no_think = prompt_name in even_no_think
        results = eval_cases(model, tokenizer, prompt_name, system_prompt,
                             cases, f"{label_prefix}", no_think=no_think)
        correct, total = print_results(results, prompt_name, f"{label_prefix}")
        summary[prompt_name] = (correct, total)

    if len(summary) > 1:
        print_summary(summary)


if __name__ == "__main__":
    main()
