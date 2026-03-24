"""Test the best prompt against real unseen speech from WhisperKit fixtures.

Usage:
    cd /Users/jud/Projects/operator
    .venv-poc/bin/python docs/hypotheses/prompt-iteration/iterate_real.py
"""

import time
import mlx_lm

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-8bit"

# Real speech from WhisperKit fixtures — model has never seen these
REAL_CASES = [
    {
        "name": "no-background-confirmation",
        "raw": 'Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and now I\'m kind of thinking, "Should we be doing this?" Like, I\'m obviously doing a bunch of transcriptions. How many transcriptions do I need to train this neural net? And then could we just be doing this, um, you know, in a background process whenever I create a transcription?',
    },
    {
        "name": "passing-very-long",
        "raw": "Well, now when we say that, should it update all docs? Or I guess the question is, like, you know, we will end up with, you know, like, we'll have these like research findings and stuff. And then maybe some of those will be proven incorrect. And the question is then what should happen to that? I think we need guardrails around agents just like updating those after the fact. You know, I think like they do, like those findings mean something. They may not be exactly correct or there may be nuance hidden in them, but I think for For someone else to come by and stamp it as like wrong, seems like the wrong approach. But maybe we stamp it as like, you know, I don't know. It may be possible to say like, that it didn't work as planned or something, I don't know. Does that make sense?",
    },
    {
        "name": "timestamp-drift",
        "raw": "I guess the question that I have is like this still feels like it's I mean I feel like we're not accounting for the fact that like the output from the large model, at least the way that I'm thinking about it, they all say the same thing in approximately the same duration. The large model has, you know, the small and medium models have like, you know, whatever, like add the absolute number of spikes above, you know, the absolute number of spikes above 0.05. Small and medium are like, you know, 0.1 percent and large is 1 percent. Like that's a big difference. I don't, like, I feel like it's the like 99 percentile whatnot. Like is that taking into account that like it's the same, It should be the same length audio even though it's a larger model.",
    },
    {
        "name": "zero-progress-cycle",
        "raw": "We should trace the last few traces. A bunch of them are not transcribing. And we should figure out what's going on in our new pipeline and then add these test cases.",
    },
    {
        "name": "confirmation-stall",
        "raw": "We should change this local copy to just be a simlink from the JStack. Well, I think we can actually delete our local copy because it will be simlinked into our clog folder.",
    },
    {
        "name": "passing-medium-long",
        "raw": "This seems right. Useful addition seems correct. I would say that the only thing that I would say is that we should be sure that if there's, if we're in AFK mode, we shouldn't be asking the user. We should use the context that the main thread has to answer the questions to the best of our ability and make make good judgments and defaults.",
    },
    {
        "name": "segment-fallback",
        "raw": "In my projects, in a, or section nine projects, but in my home, ENA folder, there's a like master schedule. I need you to find all the giant schemes and put them on the shared Judd and M calendar.",
    },
]

PROMPTS = {
    "v2-preserve-framing": {
        "system": """You clean up speech-to-text transcriptions. Remove filler words and self-corrections but keep the speaker's framing and intent. Return ONLY the cleaned text.

Examples:
Input: Um, I think we should, uh, probably go with the first option
Output: I think we should probably go with the first option.

Input: Send it to marketing, or wait no, send it to sales
Output: Send it to sales.

Input: So like, the thing is, you know, we need to figure out the deployment
Output: The thing is, we need to figure out the deployment.

Input: It's on Tuesday, actually Wednesday, yeah Wednesday at 3
Output: It's on Wednesday at 3.

Input: I think that, well, we probably should, you know, just go ahead and merge it because, like, the tests pass
Output: I think we should just go ahead and merge it because the tests pass.""",
        "user": "{raw}",
    },

    "v6-whisper-context": {
        "system": """You receive raw Whisper speech-to-text output. The speaker is dictating — they expect clean written text, not a transcript. Remove all verbal artifacts:
- Fillers: um, uh, like, you know, so, basically, I mean, right
- Self-corrections: "X, no wait, Y" → keep only Y
- False starts: incomplete phrases that get restarted
- Hedging that adds no meaning: "I think that, well, the thing is"

Keep everything else. Do not add words. Do not summarize. Return ONLY the cleaned text.""",
        "user": """Examples:
Input: Send it to the design team, or actually, the engineering team, yeah the engineering team
Output: Send it to the engineering team.

Input: I think that, well, we should probably, you know, just delete it because, like, it's not needed anymore, right? And it's just taking up space.
Output: I think we should just delete it because it's not needed anymore. It's just taking up space.

Input: {raw}
Output:""",
    },
}


def main():
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print("Loaded.\n")

    # Warmup
    mlx_lm.generate(
        model, tokenizer,
        prompt=tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        ),
        max_tokens=8, verbose=False,
    )

    for prompt_name, prompt_cfg in PROMPTS.items():
        print(f"{'='*70}")
        print(f"PROMPT: {prompt_name}")
        print(f"{'='*70}\n")

        system = prompt_cfg["system"]
        user_template = prompt_cfg["user"]

        for case in REAL_CASES:
            user_msg = user_template.format(raw=case["raw"])

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user_msg})

            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )

            t0 = time.perf_counter()
            output = mlx_lm.generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=512,
                verbose=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            output = output.strip()
            if "<think>" in output:
                import re
                output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

            raw_words = len(case["raw"].split())
            out_words = len(output.split())
            reduction = (1 - out_words / raw_words) * 100 if raw_words else 0

            print(f"  [{case['name']}] ({elapsed:.0f}ms, {raw_words}→{out_words} words, {reduction:+.0f}%)")
            print(f"    Raw:    {case['raw'][:150]}{'...' if len(case['raw']) > 150 else ''}")
            print(f"    Output: {output[:150]}{'...' if len(output) > 150 else ''}")
            print()

        print()


if __name__ == "__main__":
    main()
