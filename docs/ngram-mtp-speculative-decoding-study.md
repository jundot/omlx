# N-gram + MTP Speculative Decoding Notes

## Goal

Test whether n-gram speculation can make long roleplay generation faster in oMLX, and whether it should be combined with MTP.

## Short Answer

Yes, it helps on long repeated conversations.

Best current routing:

```text
1. Try short used-priority n-gram draft.
2. If n-gram misses, try MTP fallback.
3. If MTP fallback is not accepting enough, disable it for the rest of the request.
4. Fall back to plain target greedy when needed.
```

## Current Recommended Settings

```text
ngram_spec_enabled = true
ngram_spec_n_match = 4
ngram_spec_draft_min = 1
ngram_spec_draft_max = 2
ngram_spec_min_count = 3
ngram_spec_min_confidence = 0.8
ngram_spec_max_entries = 2048
ngram_spec_mtp_fallback = true
ngram_spec_mtp_adaptive = true
ngram_spec_mtp_min_cycles = 8
ngram_spec_mtp_min_accept_rate = 0.5
```

## Key Ideas

### 1. N-gram should be short

Long n-gram drafts caused problems.

On the 40-turn roleplay test:

| `draft_max` | Correct | Speed |
|---:|---:|---:|
| 1 | yes | 61.70 tok/s |
| 2 | yes | 72.50 tok/s |
| 4 | no | diverged |
| 8 | no | diverged |

So the default should stay small:

```text
ngram_spec_draft_max = 2
```

### 2. Used n-grams should win over frequent n-grams

Prompt frequency alone is not enough.

Example:

```text
Key: Archive keeper: The
Frequent prompt continuation: eastern aisle...
Current live continuation:  western stair...
```

If the model already used `western` in this generation, that should be prioritized over the more frequent prompt branch.

So the implementation uses:

```text
used n-gram table first
frequency table second
```

### 3. MTP helps, but not everywhere

MTP-only works on many prompts.

Example 40-turn run:

```text
Plain greedy: 46.67 tok/s
MTP-only:     53.09 tok/s
```

But n-gram helps more on repeated conversations:

```text
N-gram target fallback: 66.15 tok/s
```

MTP is best used as a fallback after n-gram misses, not as the main strategy for repeated roleplay text.

### 4. Adaptive MTP fallback is best

MTP fallback can help, but if it starts rejecting too much, it becomes overhead.

So we track MTP fallback accept rate per request.

If accept rate is too low after enough cycles, MTP fallback is disabled for the rest of the request.

## Benchmark Results

### 40-turn roleplay benchmark

Generation length: 320 tokens.

| Path | Correct | wall tok/s | decode tok/s |
|---|---:|---:|---:|
| Plain greedy | yes | 48.72 | 62.67 |
| N-gram + target fallback | yes | 67.44 | 101.33 |
| N-gram + MTP fallback | yes | 68.85 | 103.87 |
| N-gram + adaptive MTP fallback | yes | 69.61 | 104.35 |

Best result:

```text
N-gram + adaptive MTP fallback
69.61 tok/s wall throughput
104.35 tok/s decode throughput
```

### Prompt-shape matrix

| Case | Best path | Result |
|---|---|---|
| Low-repeat prose | MTP-only | small gain |
| Short repeated oath | N-gram | small gain |
| 40-turn conversation | N-gram + adaptive MTP | large gain |
| Branch-heavy repeated prompt | unsafe | speculative paths diverged |

## Example N-gram Suggestions

From the 40-turn roleplay prompt:

| Key | Suggested draft |
|---|---|
| `remember the river,` | ` the tower` |
| `the river, the` | ` tower,` |
| `river, the tower` | `, and` |
| `the tower, and` | ` the name` |
| `and the name beneath` | ` the glass` |
| `Mira: The` | ` river mark` |
| `The river mark is` | ` still cold` |

Replay stats:

```text
315 n-gram suggestion events
313 full matches
629 drafted tokens
627 accepted-prefix tokens
```

## N-gram vs MTP Overlap

Diagnostic test on the 40-turn prompt:

```text
overlap events:       110
first-token agree:     52
first-token disagree:  58
agreement rate:      47.3%
```

Meaning:

- N-gram and MTP are not redundant.
- N-gram is better at exact repeated text.
- MTP is better as a local model-based fallback.

Example agreement:

| Key | N-gram | MTP |
|---|---|---|
| `tower, and the` | ` name beneath` | ` name` |
| `The river mark is` | ` still cold` | ` still` |

Example disagreement:

| Key | N-gram | MTP |
|---|---|---|
| `Mira: The` | ` river mark` | ` name` |
| `Archive keeper:` | ` Name the` | ` The` |

## Remaining Risks

The branch-heavy repeated prompt still diverged under speculative modes.

So this is not yet a universal production-safe optimization for every prompt shape.

Safe target use case:

```text
long repeated conversation / roleplay structure
greedy decoding
short n-gram drafts
adaptive MTP fallback
```

## Conclusion

N-gram speculation is useful for long roleplay conversations because the text has repeated structure.

MTP also works, but it is better as an adaptive fallback.

The best current policy is:

```text
short used-priority n-gram first
adaptive MTP fallback second
plain target greedy fallback when MTP stops helping
```
