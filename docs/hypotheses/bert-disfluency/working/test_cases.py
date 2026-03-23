"""Test cases for BERT disfluency classification hypothesis test.

Same 12 test cases as the Qwen3.5 2B and 4B tests for direct comparison.

Each test case has:
  - name: short identifier
  - raw: the raw transcription input (as if from WhisperKit)
  - expected: hand-written cleaned output
  - category: one of filler, self-correction, ramble, clean-passthrough,
               short-command, real-fixture
"""

TEST_CASES = [
    # --- Filler removal (3) ---
    {
        "name": "filler-simple",
        "raw": "Um, I think we should, uh, go with option B",
        "expected": "I think we should go with option B.",
        "category": "filler",
    },
    {
        "name": "filler-heavy",
        "raw": (
            "So like, you know, the thing is, uh, we need to like figure"
            " this out"
        ),
        "expected": "The thing is, we need to figure this out.",
        "category": "filler",
    },
    {
        "name": "filler-with-correction",
        "raw": (
            "Yeah, I definitely wanted to use CodexExec instead of the,"
            " um, instead of the Anthropic API. Um, and now I'm kind of"
            ' thinking, should we be doing this?'
        ),
        "expected": (
            "Yeah, I definitely wanted to use CodexExec instead of the"
            " Anthropic API. And now I'm kind of thinking, should we be"
            " doing this?"
        ),
        "category": "filler",
    },

    # --- Self-correction (2) ---
    {
        "name": "correction-day",
        "raw": "Tell him Tuesday -- actually no, Wednesday works better",
        "expected": "Tell him Wednesday works better.",
        "category": "self-correction",
    },
    {
        "name": "correction-team",
        "raw": (
            "I'll send it to the product team, or wait, the engineering"
            " team, yeah the engineering team"
        ),
        "expected": "I'll send it to the engineering team.",
        "category": "self-correction",
    },

    # --- Stream-of-consciousness ramble (2) ---
    {
        "name": "ramble-research-guardrails",
        "raw": (
            "Well, now when we say that, should it update all docs? Or I"
            " guess the question is, like, you know, we will end up with,"
            " you know, like, we'll have these like research findings and"
            " stuff. And then maybe some of those will be proven incorrect."
            " And the question is then what should happen to that? I think"
            " we need guardrails around agents just like updating those"
            " after the fact. You know, I think like they do, like those"
            " findings mean something. They may not be exactly correct or"
            " there may be nuance hidden in them, but I think for For"
            " someone else to come by and stamp it as like wrong, seems"
            " like the wrong approach. But maybe we stamp it as like, you"
            " know, I don't know. It may be possible to say like, that it"
            " didn't work as planned or something, I don't know. Does that"
            " make sense?"
        ),
        "expected": (
            "Should it update all docs? We'll end up with research findings,"
            " and some of those may be proven incorrect. What should happen"
            " then? I think we need guardrails around agents updating those"
            " after the fact. Those findings mean something. They may not be"
            " exactly correct or there may be nuance hidden in them, but for"
            " someone else to come by and stamp it as wrong seems like the"
            " wrong approach. Maybe we stamp it as something like \"didn't"
            ' work as planned." Does that make sense?'
        ),
        "category": "ramble",
    },
    {
        "name": "ramble-codex-background",
        "raw": (
            "Yeah, I definitely wanted to use CodexExec instead of the, um,"
            " instead of the Anthropic API. Um, and now I'm kind of"
            ' thinking, "Should we be doing this?" Like, I\'m obviously'
            " doing a bunch of transcriptions. How many transcriptions do I"
            " need to train this neural net? And then could we just be doing"
            " this, um, you know, in a background process whenever I create"
            " a transcription?"
        ),
        "expected": (
            "I definitely wanted to use CodexExec instead of the Anthropic"
            " API. Now I'm thinking, should we be doing this? I'm doing a"
            " bunch of transcriptions. How many do I need to train this"
            " neural net? Could we just do this in a background process"
            " whenever I create a transcription?"
        ),
        "category": "ramble",
    },

    # --- Clean passthrough (2) ---
    {
        "name": "clean-meeting",
        "raw": "The meeting is at 3pm in conference room B.",
        "expected": "The meeting is at 3pm in conference room B.",
        "category": "clean-passthrough",
    },
    {
        "name": "clean-pr",
        "raw": "Please review the pull request and merge it when ready.",
        "expected": "Please review the pull request and merge it when ready.",
        "category": "clean-passthrough",
    },

    # --- Short commands (2) ---
    {
        "name": "command-timer",
        "raw": "Hey Siri set a timer for 10 minutes",
        "expected": "Set a timer for 10 minutes.",
        "category": "short-command",
    },
    {
        "name": "command-message",
        "raw": "Send a message to John",
        "expected": "Send a message to John.",
        "category": "short-command",
    },

    # --- Real fixture (1) ---
    {
        "name": "fixture-afk-mode",
        "raw": (
            "This seems right. Useful addition seems correct. I would say"
            " that the only thing that I would say is that we should be sure"
            " that if there's, if we're in AFK mode, we shouldn't be asking"
            " the user. We should use the context that the main thread has"
            " to answer the questions to the best of our ability and make"
            " make good judgments and defaults."
        ),
        "expected": (
            "This seems right. Useful addition. The only thing I would say"
            " is that if we're in AFK mode, we shouldn't be asking the"
            " user. We should use the context that the main thread has to"
            " answer the questions to the best of our ability and make good"
            " judgments and defaults."
        ),
        "category": "real-fixture",
    },
]
