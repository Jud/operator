# STT Cleanup Hypothesis Test Report

Comparison of 5 models tested for cleaning up raw speech-to-text output.
All models were evaluated on the same 12 test cases covering fillers,
self-corrections, rambling speech, clean passthrough, short commands,
and real WhisperKit fixture text.

## Summary

| Model | Type | Params | Filler Removal | Self-Correction | Meaning Preservation | Verdict |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | Generative LLM | 2B | — | — | 6/12 passed | **Disproven** |
| Qwen3.5-4B | Generative LLM | 4B | — | — | 7/12 passed | **Disproven** |
| ModernBERT-base | Token classifier | 150M | — | — | 5/12 passed | **Disproven** |
| ModernBERT-large | Token classifier | 396M | — | — | 5/12 passed | **Disproven** |
| CoEdit-large | Text editor (T5) | 770M | — | — | 7/12 passed | **Disproven** |

## Per-Case Results

### filler-simple

**Input:**
> Um, I think we should, uh, go with option B

**Expected:**
> I think we should go with option B.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | I think we should go with option B | 97% | 320ms | pass |
| Qwen3.5-4B | I think we should go with option B. | 100% | 602ms | pass |
| ModernBERT-base | , I think we should,, go with option B | 87% | 541ms | pass |
| ModernBERT-large | , I think we should,, go with option B | 87% | 198ms | pass |
| CoEdit-large | Um, I think we should, uh, go with option B. | 89% | 7288ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Um, I think we should, uh, go with option B. | 89% |
| `Fix the grammar in this text:` | Um, I think we should, uh, go with option B. | 89% |
| `Remove disfluencies from this text:` | Um, I think we should, uh, go with option B. | 89% |
| `Fix the grammar and remove filler words:` | Um, I think we should, uh, go with option B. | 89% |

**BERT token labels:**

_ModernBERT-base:_
```
Um[FP] ,[O]  I[O]  think[O]  we[O]  should[O] ,[O]  uh[FP] ,[O]  go[O]  with[O]  option[O]  B[O]
```

_ModernBERT-large:_
```
Um[FP] ,[O]  I[O]  think[O]  we[O]  should[O] ,[O]  uh[FP] ,[O]  go[O]  with[O]  option[O]  B[O]
```

### filler-heavy

**Input:**
> So like, you know, the thing is, uh, we need to like figure this out

**Expected:**
> The thing is, we need to figure this out.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | So, the thing is, we need to figure this out | 89% | 347ms | **FAIL** |
| Qwen3.5-4B | The thing is, we need to figure this out. | 100% | 633ms | pass |
| ModernBERT-base | So like you know, the thing is,, we need to like figure this out | 61% | 119ms | **FAIL** |
| ModernBERT-large | So like, you know, the thing is,, we need to like figure this out | 60% | 74ms | **FAIL** |
| CoEdit-large | So, you know, the thing is, uh, we need to like figure this out. | 78% | 1075ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | So, you know, the thing is, uh, we need to like figure this out. | 78% |
| `Fix the grammar in this text:` | So, you know, the thing is, uh, we need to like figure this out. | 78% |
| `Remove disfluencies from this text:` | So, you know, the thing is, uh, we need to like figure this out. | 78% |
| `Fix the grammar and remove filler words:` | So, you know, the thing is, uh, we need to like figure this out. | 78% |

**BERT token labels:**

_ModernBERT-base:_
```
So[O]  like[O] ,[RV]  you[O]  know[O] ,[O]  the[O]  thing[O]  is[O] ,[O]  uh[FP] ,[O]  we[O]  need[O]  to[O]  like[O]  figure[O]  this[O]  out[O]
```

_ModernBERT-large:_
```
So[O]  like[O] ,[O]  you[O]  know[O] ,[O]  the[O]  thing[O]  is[O] ,[O]  uh[FP] ,[O]  we[O]  need[O]  to[O]  like[O]  figure[O]  this[O]  out[O]
```

### filler-with-correction

**Input:**
> Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and now I'm kind of thinking, should we be doing this?

**Expected:**
> Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind of thinking, should we be doing this?

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind of thinkin… | 100% | 517ms | pass |
| Qwen3.5-4B | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind of thinkin… | 100% | 971ms | pass |
| ModernBERT-base | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API., and now I 'm kind of think… | 98% | 77ms | pass |
| ModernBERT-large | Yeah, I definitely wanted use CodexExec,, instead of the Anthropic API., and now I 'm kind of thinki… | 94% | 32ms | pass |
| CoEdit-large | Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and… | 91% | 2070ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and… | 91% |
| `Fix the grammar in this text:` | Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and… | 91% |
| `Remove disfluencies from this text:` | Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and… | 91% |
| `Fix the grammar and remove filler words:` | Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and… | 91% |

**BERT token labels:**

_ModernBERT-base:_
```
Yeah[O] ,[O]  I[O]  definitely[O]  wanted[O]  to[O]  use[O]  CodexExec[O]  instead[RV]  of[RV]  the[RV] ,[RV]  um[FP] ,[RV]  instead[O]  of[O]  the[O]  Anthropic[O]  API[O] .[O]  Um[FP] ,[O]  and[O]  now[O]  I[O] 'm[O]  kind[O]  of[O]  thinking[O] ,[O]  should[O]  we[O]  be[O]  doing[O]  this[O] ?[O]
```

_ModernBERT-large:_
```
Yeah[O] ,[O]  I[O]  definitely[O]  wanted[O]  to[RP]  use[O]  CodexExec[O]  instead[RP]  of[RP]  the[RP] ,[O]  um[FP] ,[O]  instead[O]  of[O]  the[O]  Anthropic[O]  API[O] .[O]  Um[FP] ,[O]  and[O]  now[O]  I[O] 'm[O]  kind[O]  of[O]  thinking[O] ,[O]  should[O]  we[O]  be[O]  doing[O]  this[O] ?[O]
```

### correction-day

**Input:**
> Tell him Tuesday -- actually no, Wednesday works better

**Expected:**
> Tell him Wednesday works better.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Tell him Tuesday -- actually no, Wednesday works better | 55% | 340ms | **FAIL** |
| Qwen3.5-4B | Tell him Tuesday; actually, no, Wednesday works better. | 58% | 652ms | **FAIL** |
| ModernBERT-base | Tell him Tuesday-- no, Wednesday works better | 67% | 444ms | **FAIL** |
| ModernBERT-large | Tell him Tuesday-- actually no, Wednesday works better | 56% | 197ms | **FAIL** |
| CoEdit-large | Tell him Tuesday -- actually, no, Wednesday works better. | 72% | 598ms | **FAIL** |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Tell him Tuesday -- actually, no, Wednesday works better. | 72% |
| `Fix the grammar in this text:` | Tell him Tuesday -- actually, no, Wednesday works better. | 72% |
| `Remove disfluencies from this text:` | Tell him Tuesday -- actually, no, Wednesday works better. | 72% |
| `Fix the grammar and remove filler words:` | Tell him Tuesday -- actually, no, Wednesday works better. | 72% |

**BERT token labels:**

_ModernBERT-base:_
```
Tell[O]  him[O]  Tuesday[O]  --[O]  actually[RV]  no[O] ,[O]  Wednesday[O]  works[O]  better[O]
```

_ModernBERT-large:_
```
Tell[O]  him[O]  Tuesday[O]  --[O]  actually[O]  no[O] ,[O]  Wednesday[O]  works[O]  better[O]
```

### correction-team

**Input:**
> I'll send it to the product team, or wait, the engineering team, yeah the engineering team

**Expected:**
> I'll send it to the engineering team.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | I'll send it to the product team, or wait, the engineering team, yeah the engineering team | 40% | 414ms | **FAIL** |
| Qwen3.5-4B | I'll send it to the product team, or wait, the engineering team, yeah, the engineering team. | 40% | 791ms | **FAIL** |
| ModernBERT-base | I 'll send it to the product team, or wait,, yeah the team | 47% | 50ms | **FAIL** |
| ModernBERT-large | I 'll send it to the product team, or wait,, yeah the engineering team | 50% | 27ms | **FAIL** |
| CoEdit-large | I'll send it to the product team, or wait for the engineering team, yeah, the engineering team. | 56% | 762ms | **FAIL** |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | I'll send it to the product team, or wait for the engineering team, yeah, the engineering team. | 56% |
| `Fix the grammar in this text:` | I'll send it to the product team, or wait for the engineering team, yeah, the engineering team. | 56% |
| `Remove disfluencies from this text:` | I'll send it to the product team, or wait for the engineering team, yeah, the engineering team. | 56% |
| `Fix the grammar and remove filler words:` | I'll send it to the product team, or wait for the engineering team, yeah, the engineering team. | 56% |

**BERT token labels:**

_ModernBERT-base:_
```
I[O] 'll[O]  send[O]  it[O]  to[O]  the[O]  product[O]  team[O] ,[O]  or[O]  wait[O] ,[O]  the[RP]  engineering[RP]  team[RP] ,[O]  yeah[O]  the[O]  engineering[RP]  team[O]
```

_ModernBERT-large:_
```
I[O] 'll[O]  send[O]  it[O]  to[O]  the[O]  product[O]  team[O] ,[O]  or[O]  wait[O] ,[O]  the[RP]  engineering[RP]  team[RP] ,[O]  yeah[O]  the[O]  engineering[O]  team[O]
```

### ramble-research-guardrails

**Input:**
> Well, now when we say that, should it update all docs? Or I guess the question is, like, you know, we will end up with, you know, like, we'll have these like research findings and stuff. And then maybe some of those will be proven incorrect. And the question is then what should happen to that? I think we need guardrails around agents just like updating those after the fact. You know, I think like they do, like those findings mean something. They may not be exactly correct or there may be nuance hidden in them, but I think for For someone else to come by and stamp it as like wrong, seems like the wrong approach. But maybe we stamp it as like, you know, I don't know. It may be possible to say like, that it didn't work as planned or something, I don't know. Does that make sense?

**Expected:**
> Should it update all docs? We'll end up with research findings, and some of those may be proven incorrect. What should happen then? I think we need guardrails around agents updating those after the fact. Those findings mean something. They may not be exactly correct or there may be nuance hidden in them, but for someone else to come by and stamp it as wrong seems like the wrong approach. Maybe we stamp it as something like "didn't work as planned." Does that make sense?

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Well, now when we say that, should it update all docs? Or I guess the question is, like, you know, w… | 59% | 1698ms | **FAIL** |
| Qwen3.5-4B | Well, now when we say that, should it update all docs? Or, I guess the question is, we will end up w… | 66% | 3112ms | **FAIL** |
| ModernBERT-base | say that, should it update all docs? Or I you know,, we 'll these like research and. And then maybe … | 63% | 507ms | **FAIL** |
| ModernBERT-large | Well, now when we say that, should it update all docs? Or I guess the question is, like, you know, w… | 56% | 204ms | **FAIL** |
| CoEdit-large | Well, now when we say that, should it update all the docs? Or I guess the question is, like, you kno… | 1% | 13524ms | **FAIL** |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Well, now when we say that, should it update all the docs? Or I guess the question is, like, you kno… | 1% |
| `Fix the grammar in this text:` | Well, now when we say that, should it update all the docs? Or I guess the question is, like, you kno… | 1% |
| `Remove disfluencies from this text:` | Well, now when we say that, should it update all the docs? Or I guess the question is, like, you kno… | 1% |
| `Fix the grammar and remove filler words:` | Well, now when we say that, should it update all the docs? Or I guess the question is, like, you kno… | 1% |

**BERT token labels:**

_ModernBERT-base:_
```
Well[RV] ,[RV]  now[RV]  when[RV]  we[RV]  say[O]  that[O] ,[O]  should[O]  it[O]  update[O]  all[O]  docs[O] ?[O]  Or[O]  I[O]  guess[RV]  the[RV]  question[RV]  is[RV] ,[RV]  like[RV] ,[RV]  you[RV]  know[RV] ,[RV]  we[RV]  will[RV]  end[RV]  up[RV]  with[RV] ,[RV]  you[O]  know[O] ,[O]  like[RV] ,[O]  we[O] 'll[O]  have[RV]  these[O]  like[O]  research[O]  findings[RV]  and[O]  stuff[RV] .[O]  And[O]  then[O]  maybe[O]  some[O]  of[O]  those[O]  will[O]  be[O]  proven[O]  incorrect[O] .[O]  A
```

_ModernBERT-large:_
```
Well[O] ,[O]  now[O]  when[O]  we[O]  say[O]  that[O] ,[O]  should[O]  it[O]  update[O]  all[O]  docs[O] ?[O]  Or[O]  I[O]  guess[O]  the[O]  question[O]  is[O] ,[O]  like[O] ,[O]  you[O]  know[O] ,[O]  we[O]  will[O]  end[O]  up[O]  with[O] ,[O]  you[O]  know[O] ,[O]  like[O] ,[O]  we[O] 'll[O]  have[O]  these[O]  like[O]  research[O]  findings[O]  and[O]  stuff[O] .[O]  And[O]  then[O]  maybe[O]  some[O]  of[O]  those[O]  will[O]  be[O]  proven[O]  incorrect[O] .[O]  And[O]  the[O]  question[O
```

### ramble-codex-background

**Input:**
> Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and now I'm kind of thinking, "Should we be doing this?" Like, I'm obviously doing a bunch of transcriptions. How many transcriptions do I need to train this neural net? And then could we just be doing this, um, you know, in a background process whenever I create a transcription?

**Expected:**
> I definitely wanted to use CodexExec instead of the Anthropic API. Now I'm thinking, should we be doing this? I'm doing a bunch of transcriptions. How many do I need to train this neural net? Could we just do this in a background process whenever I create a transcription?

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind of thinkin… | 80% | 847ms | **FAIL** |
| Qwen3.5-4B | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind of thinkin… | 80% | 1722ms | **FAIL** |
| ModernBERT-base | Yeah, I definitely wanted to use CodexExec., and now I 'm kind of thinking, " Should we be doing thi… | 70% | 131ms | **FAIL** |
| ModernBERT-large | Yeah, I definitely wanted to use CodexExec,, instead of the Anthropic API., and now I 'm kind thinki… | 77% | 109ms | **FAIL** |
| CoEdit-large | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. Um, and now I'm kind of thi… | 22% | 30992ms | **FAIL** |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. Um, and now I'm kind of thi… | 22% |
| `Fix the grammar in this text:` | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. Um, and now I'm kind of thi… | 22% |
| `Remove disfluencies from this text:` | Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. Um, and now I'm kind of thi… | 22% |
| `Fix the grammar and remove filler words:` | Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and… | 21% |

**BERT token labels:**

_ModernBERT-base:_
```
Yeah[O] ,[O]  I[O]  definitely[O]  wanted[O]  to[O]  use[O]  CodexExec[O]  instead[RV]  of[RV]  the[RV] ,[RV]  um[FP] ,[RV]  instead[RV]  of[RV]  the[RV]  Anthropic[RV]  API[RV] .[O]  Um[FP] ,[O]  and[O]  now[O]  I[O] 'm[O]  kind[O]  of[O]  thinking[O] ,[O]  "[O] Should[O]  we[O]  be[O]  doing[O]  this[O] ?"[O]  Like[O] ,[O]  I[O] 'm[O]  obviously[O]  doing[O]  a[O]  bunch[O]  of[O]  transcriptions[O] .[O]  How[O]  many[O]  transcriptions[O]  do[O]  I[O]  need[O]  to[O]  train[O]  this[O]  neura
```

_ModernBERT-large:_
```
Yeah[O] ,[O]  I[O]  definitely[O]  wanted[O]  to[O]  use[O]  CodexExec[O]  instead[RP]  of[RP]  the[RP] ,[O]  um[FP] ,[O]  instead[O]  of[O]  the[O]  Anthropic[O]  API[O] .[O]  Um[FP] ,[O]  and[O]  now[O]  I[O] 'm[O]  kind[O]  of[RP]  thinking[O] ,[O]  "[O] Should[O]  we[O]  be[O]  doing[O]  this[O] ?"[O]  Like[O] ,[O]  I[O] 'm[O]  obviously[O]  doing[O]  a[O]  bunch[O]  of[O]  transcriptions[O] .[O]  How[O]  many[O]  transcriptions[O]  do[O]  I[O]  need[O]  to[O]  train[O]  this[O]  neural[O]  
```

### clean-meeting

**Input:**
> The meeting is at 3pm in conference room B.

**Expected:**
> The meeting is at 3pm in conference room B.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | The meeting is at 3pm in conference room B. | 100% | 349ms | pass |
| Qwen3.5-4B | The meeting is at 3pm in conference room B. | 100% | 738ms | pass |
| ModernBERT-base | The meeting is at 3 pm in conference room B. | 98% | 465ms | **FAIL** |
| ModernBERT-large | The meeting is at 3 pm in conference room B. | 98% | 196ms | **FAIL** |
| CoEdit-large | The meeting is at 3pm in conference room B. | 100% | 524ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | The meeting will be at 3pm in conference room B. | 92% |
| `Fix the grammar in this text:` | The meeting is at 3pm in conference room B. | 100% |
| `Remove disfluencies from this text:` | The meeting is at 3pm in conference room B. | 100% |
| `Fix the grammar and remove filler words:` | The meeting is at 3pm in conference room B. | 100% |

**BERT token labels:**

_ModernBERT-base:_
```
The[O]  meeting[O]  is[O]  at[O]  3[O] pm[O]  in[O]  conference[O]  room[O]  B[O] .[O]
```

_ModernBERT-large:_
```
The[O]  meeting[O]  is[O]  at[O]  3[O] pm[O]  in[O]  conference[O]  room[O]  B[O] .[O]
```

### clean-pr

**Input:**
> Please review the pull request and merge it when ready.

**Expected:**
> Please review the pull request and merge it when ready.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Please review the pull request and merge it when ready. | 100% | 396ms | pass |
| Qwen3.5-4B | Please review the pull request and merge it when ready. | 100% | 662ms | pass |
| ModernBERT-base | Please review the pull request and merge it when ready. | 100% | 15ms | pass |
| ModernBERT-large | Please review the pull request and merge it when ready. | 100% | 17ms | pass |
| CoEdit-large | Please review the pull request and merge it when ready. | 100% | 444ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Please review the pull request and merge it when ready. | 100% |
| `Fix the grammar in this text:` | Please review the pull request and merge it when ready. | 100% |
| `Remove disfluencies from this text:` | Please review the pull request and merge it when ready. | 100% |
| `Fix the grammar and remove filler words:` | Please review the pull request and merge it when ready. | 100% |

**BERT token labels:**

_ModernBERT-base:_
```
Please[O]  review[O]  the[O]  pull[O]  request[O]  and[O]  merge[O]  it[O]  when[O]  ready[O] .[O]
```

_ModernBERT-large:_
```
Please[O]  review[O]  the[O]  pull[O]  request[O]  and[O]  merge[O]  it[O]  when[O]  ready[O] .[O]
```

### command-timer

**Input:**
> Hey Siri set a timer for 10 minutes

**Expected:**
> Set a timer for 10 minutes.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Hey Siri set a timer for 10 minutes | 71% | 334ms | **FAIL** |
| Qwen3.5-4B | Hey Siri, set a timer for 10 minutes. | 73% | 662ms | **FAIL** |
| ModernBERT-base | Hey Siri set a timer for 10 minutes | 71% | 457ms | **FAIL** |
| ModernBERT-large | Hey Siri set a timer for 10 minutes | 71% | 182ms | **FAIL** |
| CoEdit-large | Hey Siri, set a timer for 10 minutes. | 84% | 429ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Hey Siri, set a timer for 10 minutes. | 84% |
| `Fix the grammar in this text:` | Hey Siri, set a timer for 10 minutes. | 84% |
| `Remove disfluencies from this text:` | Hey Siri, set a timer for 10 minutes. | 84% |
| `Fix the grammar and remove filler words:` | Hey Siri, set a timer for 10 minutes. | 84% |

**BERT token labels:**

_ModernBERT-base:_
```
Hey[O]  Siri[O]  set[O]  a[O]  timer[O]  for[O]  10[O]  minutes[O]
```

_ModernBERT-large:_
```
Hey[O]  Siri[O]  set[O]  a[O]  timer[O]  for[O]  10[O]  minutes[O]
```

### command-message

**Input:**
> Send a message to John

**Expected:**
> Send a message to John.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | Send a message to John | 96% | 300ms | pass |
| Qwen3.5-4B | Send a message to John | 96% | 546ms | pass |
| ModernBERT-base | Send a message to John | 96% | 351ms | pass |
| ModernBERT-large | Send a message to John | 96% | 239ms | pass |
| CoEdit-large | Send a message to John | 98% | 548ms | pass |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | Send a message to John | 98% |
| `Fix the grammar in this text:` | Send a message to John | 98% |
| `Remove disfluencies from this text:` | Send a message to John | 98% |
| `Fix the grammar and remove filler words:` | Send a message to John | 98% |

**BERT token labels:**

_ModernBERT-base:_
```
Send[O]  a[O]  message[O]  to[O]  John[O]
```

_ModernBERT-large:_
```
Send[O]  a[O]  message[O]  to[O]  John[O]
```

### fixture-afk-mode

**Input:**
> This seems right. Useful addition seems correct. I would say that the only thing that I would say is that we should be sure that if there's, if we're in AFK mode, we shouldn't be asking the user. We should use the context that the main thread has to answer the questions to the best of our ability and make make good judgments and defaults.

**Expected:**
> This seems right. Useful addition. The only thing I would say is that if we're in AFK mode, we shouldn't be asking the user. We should use the context that the main thread has to answer the questions to the best of our ability and make good judgments and defaults.

| Model | Output | Similarity | Latency | Pass |
|---|---|---|---|---|
| Qwen3.5-2B | This seems right. Useful addition seems correct. I would say that the only thing I would say is that… | 83% | 808ms | pass |
| Qwen3.5-4B | This seems right. Useful addition seems correct. I would say that the only thing I would add is that… | 82% | 1774ms | pass |
| ModernBERT-base | This seems. I would say that the only thing that I would say is that we should be sure that there, i… | 81% | 163ms | pass |
| ModernBERT-large | This seems right. Useful addition seems correct. I would say that the only thing that I would say is… | 80% | 89ms | pass |
| CoEdit-large | This seems right. A useful addition seems correct. I would say that the only thing that I would say … | 18% | 4596ms | **FAIL** |

**CoEdit prompt comparison:**

| Prompt | Output | Similarity |
|---|---|---|
| `Make this text fluent:` | This seems right. A useful addition seems correct. I would say that the only thing that I would say … | 18% |
| `Fix the grammar in this text:` | This seems right. A useful addition seems correct. I would say that the only thing that I would say … | 18% |
| `Remove disfluencies from this text:` | This seems right. A useful addition seems correct. I would say that the only thing that I would say … | 18% |
| `Fix the grammar and remove filler words:` | This seems right. A useful addition seems correct. I would say that the only thing that I would say … | 18% |

**BERT token labels:**

_ModernBERT-base:_
```
This[O]  seems[RV]  right[RV] .[RV]  Useful[RV]  addition[RV]  seems[O]  correct[RV] .[O]  I[O]  would[O]  say[O]  that[O]  the[O]  only[O]  thing[O]  that[O]  I[O]  would[O]  say[O]  is[O]  that[O]  we[O]  should[O]  be[O]  sure[O]  that[O]  if[RV]  there[O] 's[RV] ,[O]  if[O]  we[O] 're[O]  in[O]  AFK[O]  mode[O] ,[O]  we[O]  shouldn[O] 't[O]  be[O]  asking[O]  the[O]  user[O] .[O]  We[O]  should[O]  use[O]  the[O]  context[O]  that[O]  the[O]  main[O]  thread[O]  has[O]  to[O]  answer[O]  the
```

_ModernBERT-large:_
```
This[O]  seems[O]  right[O] .[O]  Useful[O]  addition[O]  seems[O]  correct[O] .[O]  I[O]  would[O]  say[O]  that[O]  the[O]  only[O]  thing[O]  that[O]  I[O]  would[O]  say[O]  is[O]  that[O]  we[O]  should[O]  be[O]  sure[O]  that[O]  if[RV]  there[RV] 's[RV] ,[RV]  if[O]  we[O] 're[O]  in[O]  AFK[O]  mode[O] ,[O]  we[O]  shouldn[O] 't[O]  be[O]  asking[O]  the[O]  user[O] .[O]  We[O]  should[O]  use[O]  the[O]  context[O]  that[O]  the[O]  main[O]  thread[O]  has[O]  to[O]  answer[O]  the[O] 
```

## System Prompts / Instructions Used

### Qwen3.5-2B

```
You are a speech-to-text cleanup tool. Your job is to transform raw voice transcriptions into clean, polished written text.

Rules:
1. Remove filler words: um, uh, like, you know, so, actually, basically
2. Resolve self-corrections: keep only the final intended version
3. Fix grammar, capitalization, and punctuation
4. Preserve the speaker's meaning exactly -- never add content
5. Output ONLY the cleaned text, no explanations, no markdown, no quotes
6. If the input is already clean, return it un
```

### Qwen3.5-4B

```
You are a speech-to-text cleanup tool. Your job is to transform raw voice transcriptions into clean, polished written text.

Rules:
1. Remove filler words: um, uh, like, you know, so, actually, basically
2. Resolve self-corrections: keep only the final intended version
3. Fix grammar, capitalization, and punctuation
4. Preserve the speaker's meaning exactly -- never add content
5. Output ONLY the cleaned text, no explanations, no markdown, no quotes
6. If the input is already clean, return it un
```

### CoEdit-large

CoEdit uses instruction prefixes prepended to input text:

1. `Make this text fluent: <input>`
2. `Fix the grammar in this text: <input>`
3. `Remove disfluencies from this text: <input>`
4. `Fix the grammar and remove filler words: <input>`

---
*Generated from hypothesis test results in `docs/hypotheses/`*