"""Prompt variants v29-v45 for routing eval.

Import and merge into PROMPTS dict in eval.py.
"""

# Shared session context blocks used across prompts
KEYWORDS_BLOCK = (
    "SESSION 1: ui-shell\n"
    "  Topics: loading, dashboard, settings, cards, mobile, layout, CSS, skeleton, responsive\n"
    "  Files: LoadingCard.tsx, SettingsRoute.tsx, dashboard.css\n"
    "  Status: fixing loading states and card layout on settings page\n\n"
    "SESSION 2: profile-api\n"
    "  Topics: profile, endpoint, timeout, SQL, query, connection pool, REST, preferences, API\n"
    "  Files: profile.py, user_profile.py, profile.sql\n"
    "  Status: fixing slow query and adding preferences endpoint\n\n"
    "SESSION 3: staging-infra\n"
    "  Topics: VPC, staging, Terraform, S3, bucket, IAM, policy, subnet, deploy, rollout\n"
    "  Files: main.tf, modules/vpc, modules/iam_policy\n"
    "  Status: fixing VPC rollout and updating IAM and S3 config"
)

FILES_ONLY_BLOCK = (
    "SESSION 1: ui-shell\n"
    "  LoadingCard.tsx, SettingsRoute.tsx, dashboard.css\n\n"
    "SESSION 2: profile-api\n"
    "  profile.py, user_profile.py, profile.sql\n\n"
    "SESSION 3: staging-infra\n"
    "  main.tf, modules/vpc, modules/iam_policy"
)

LAST_MSG_BLOCK = (
    'SESSION 1: ui-shell\n'
    '  "I updated LoadingCard.tsx, SettingsRoute.tsx, and dashboard.css; '
    'next I am wiring the pending query state and skeleton layout."\n\n'
    'SESSION 2: profile-api\n'
    '  "I traced the slowdown to profile.sql and the connection pool, '
    'and I am adding the preferences endpoint in profile.py."\n\n'
    'SESSION 3: staging-infra\n'
    '  "I am updating the private subnet module, the IAM policy document, '
    'and the S3 logging resources in staging Terraform."'
)

NOT_CONFIDENT_EXAMPLES = (
    '"fix the tests" → not confident\n'
    '"add error handling" → not confident\n'
    '"how is it going" → not confident\n'
    '"try again" → not confident\n'
    '"that looks wrong" → not confident\n'
    '"undo that" → not confident\n'
    '"the build is broken" → not confident\n'
    '"ship it" → not confident\n'
    '"commit what you have" → not confident\n'
    '"set up a CDN" → not confident'
)


def build_prompts():
    P = {}

    # ── v29: Decision tree — force the model to check generic first ──
    P["v29"] = (
        f"Route the voice command. Follow these steps:\n\n"
        f"STEP 1: Is the message generic? (could apply to any coding project)\n"
        f"  Examples of generic: fix the tests, add error handling, the build is broken, "
        f"how is it going, try again, undo that, ship it, commit, that looks wrong\n"
        f"  If YES → {{\"session\":\"\",\"confident\":false}}\n\n"
        f"STEP 2: Does the message match exactly ONE session's work?\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"  If YES → {{\"session\":\"<name>\",\"confident\":true}}\n"
        f"  If matches ZERO or MULTIPLE → {{\"session\":\"\",\"confident\":false}}\n\n"
        f"JSON only."
    )

    # ── v30: v29 + no-think ──
    P["v30"] = P["v29"]

    # ── v31: Confidence scoring — rate 1-10, threshold at 8 ──
    P["v31"] = (
        f"Rate how confident you are that this voice command belongs to one specific session.\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"Score 8-10: The message clearly relates to one session's topics or files.\n"
        f"Score 1-7: The message is generic, vague, or could match multiple sessions.\n\n"
        f"Examples:\n"
        f'  "the cards are still jumping" → score 9, ui-shell (matches loading/layout work)\n'
        f'  "is the query faster" → score 9, profile-api (matches SQL work)\n'
        f'  "fix the tests" → score 2, no match (all sessions have tests)\n'
        f'  "try again" → score 1, no match (conversational)\n'
        f'  "set up a CDN" → score 4, unclear (not in any session\'s current work)\n\n'
        f"If score >= 8: {{\"session\":\"<name>\",\"confident\":true}}\n"
        f"If score < 8: {{\"session\":\"\",\"confident\":false}}\n"
        f"JSON only."
    )

    # ── v32: v31 + no-think ──
    P["v32"] = P["v31"]

    # ── v33: Contrastive examples — show WHY something matches one and not others ──
    P["v33"] = (
        f"Match the voice command to the session doing related work.\n"
        f"Only confident when the match is clear. When in doubt → not confident.\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"EXAMPLES with reasoning:\n"
        f'  "center that div" → ui-shell ✓ (CSS/layout relates to dashboard.css work)\n'
        f'  "the response is slow" → profile-api ✓ (performance relates to SQL/endpoint timeout)\n'
        f'  "the deploy failed" → staging-infra ✓ (deploy relates to Terraform/rollout work)\n'
        f'  "someone can see other users data" → profile-api ✓ (user data relates to profile/API)\n'
        f'  "the certificate is expiring" → NOT CONFIDENT (not in any session\'s current work)\n'
        f'  "add a CDN" → NOT CONFIDENT (could be infra or frontend)\n'
        f'  "fix the tests" → NOT CONFIDENT (every session has tests)\n'
        f'  "try again" → NOT CONFIDENT (conversational, not about work)\n'
        f'  "that looks wrong" → NOT CONFIDENT (no indication which session)\n\n'
        f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    # ── v34: v33 + no-think ──
    P["v34"] = P["v33"]

    # ── v35: Files-only context — minimal, just file names ──
    P["v35"] = (
        f"Match the voice command to the session working on related files.\n"
        f"Only confident when the message relates to one session's files. Otherwise not confident.\n\n"
        f"{FILES_ONLY_BLOCK}\n\n"
        f"Not confident when generic: fix tests, add error handling, try again, how is it going, undo, ship it, that looks wrong\n\n"
        f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    # ── v36: v35 + no-think ──
    P["v36"] = P["v35"]

    # ── v37: Last assistant message verbatim — most realistic ──
    P["v37"] = (
        f"Each session has an AI agent. Here is what each agent last said:\n\n"
        f"{LAST_MSG_BLOCK}\n\n"
        f"Route the user's voice command to the session whose agent is doing the most related work.\n"
        f"If the command is generic or doesn't relate to any agent's work, set confident=false.\n\n"
        f"Generic examples (always not confident): fix the tests, try again, how is it going, "
        f"that looks wrong, undo that, ship it, the build is broken, add error handling\n\n"
        f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    # ── v38: v37 + no-think ──
    P["v38"] = P["v37"]

    # ── v39: Heavy not-confident ratio — 10 not-confident examples, 2 confident ──
    P["v39"] = (
        f"Match the voice command to a session. When in doubt, say not confident.\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"CONFIDENT (only when clearly about one session):\n"
        f'  "the cards are jumping" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
        f'  "the query is slow" → {{\"session\":\"profile-api\",\"confident\":true}}\n\n'
        f"NOT CONFIDENT (generic, vague, conversational, or ambiguous):\n"
        f'{NOT_CONFIDENT_EXAMPLES}\n\n'
        f"Default to not confident. JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    # ── v40: v39 + no-think ──
    P["v40"] = P["v39"]

    # ── v41: Top-3 ranked output instead of binary ──
    P["v41"] = (
        f"Rank which session this voice command most likely belongs to.\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"Reply with the top match and your confidence (high/low).\n"
        f"If the message is generic or conversational, confidence is low.\n\n"
        f'Examples:\n'
        f'  "the cards are jumping" → {{\"session\":\"ui-shell\",\"confident\":true}}\n'
        f'  "fix the query" → {{\"session\":\"profile-api\",\"confident\":true}}\n'
        f'  "fix the tests" → {{\"session\":\"\",\"confident\":false}}\n'
        f'  "try again" → {{\"session\":\"\",\"confident\":false}}\n\n'
        f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    # ── v42: v41 + no-think ──
    P["v42"] = P["v41"]

    # ── v43: Combined best — v27 structure + v33 contrastive + v29 decision tree ──
    P["v43"] = (
        f"Route the voice command. A wrong guess is much worse than asking.\n\n"
        f"FIRST: Is this generic or conversational? If yes → not confident.\n"
        f"  Generic: fix tests, add error handling, the build is broken, try again, "
        f"undo that, ship it, how is it going, that looks wrong, commit, revert\n\n"
        f"THEN: Does it match exactly one session's current work?\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"MATCH examples:\n"
        f'  "center that div" → ui-shell (CSS/layout)\n'
        f'  "the response is slow" → profile-api (query/endpoint)\n'
        f'  "the deploy keeps failing" → staging-infra (rollout/Terraform)\n\n'
        f"NO MATCH examples:\n"
        f'  "the certificate is expiring" → not confident (not in any session\'s work)\n'
        f'  "set up a CDN" → not confident (could be any session)\n'
        f'  "add a forgot password flow" → not confident (no session working on auth)\n\n'
        f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    # ── v44: v43 + no-think ──
    P["v44"] = P["v43"]

    # ── v45: The "ultra-conservative" — bias hard toward not confident ──
    P["v45"] = (
        f"Route ONLY if you are very sure. Default is not confident.\n\n"
        f"{KEYWORDS_BLOCK}\n\n"
        f"Set confident=true ONLY when the message directly mentions a topic or file "
        f"that belongs to exactly one session. In all other cases, set confident=false.\n\n"
        f"confident=true examples:\n"
        f'  "fix the dashboard CSS" → ui-shell (dashboard.css is in this session)\n'
        f'  "the SQL query is still slow" → profile-api (SQL/query is in this session)\n'
        f'  "terraform plan failed" → staging-infra (Terraform is in this session)\n\n'
        f"confident=false examples (this is the default):\n"
        f'{NOT_CONFIDENT_EXAMPLES}\n'
        f'  "the certificate is expiring" → not confident\n'
        f'  "add a CDN" → not confident\n'
        f'  "someone can see other users data" → not confident\n'
        f'  "add a forgot password flow" → not confident\n'
        f'  "the docker container wont start" → not confident\n\n'
        f"JSON only: {{\"session\":\"<name or empty>\",\"confident\":true|false}}"
    )

    return P
