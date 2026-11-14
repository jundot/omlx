# Decision models

A decision model answers typed questions about a state in a single forward pass.
It is a backbone encoder plus last-token pooling and a small FP32 readout head
over an alphabet of answer codes, so it generates no text: the answer is a
probability per option. oMLX serves them on `/v1/systemone` (named, typed
questions) and `/v1/decisions` (decision objects), never on
`/v1/chat/completions`.

Because the readout replaces the `lm_head` and the attention is non-causal,
loading such a checkpoint with stock `mlx_lm.load` / `mlx_vlm.load` answers
plausibly but wrong: the causal mask hides the question from the state tokens it
was trained to read, and the sampled token comes from the vocabulary head
instead of the readout.

## Supported families

Detection is by checkpoint files, not by directory name — see
`decision_kind()` in `omlx/model_discovery.py`.

| Kind | Family | Detection | Notes |
|------|--------|-----------|-------|
| `clef` | Clef, Clef-Flash | `joint_head.safetensors` + `joint_head_config.json` | `truncate` applies |
| `openjev` | OpenJev | `helper/shim.py`, or the directory name | plain Qwen3.5 backbone |
| `d3` | Decision 3.0 (d3, d3-flash, d3-mini, d3-nano, d3-lite, d3-edge) | `readout.safetensors` + `decision_config.json` | reads images and video |
| `d1` | LiquidAI d1 | `D1Model` in `architectures` on an LFM2-VL / LFM3-VL backbone | reads images |
| `pplx_decider` | Perplexity Decider v1.3 (flash, 27B) | `readout.safetensors` + `joint_config.json` with `template: joint-chunked-readout-v1` | chunked joint protocol, 262k context, reads images |

Each family keeps its own prompt protocol, pooling rule and answer codes inside
its model class in `omlx/models/`; the engine only drives the shared
encode → readout → answer pipeline.

## Request

```bash
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "pplx-decider-v1.3",
        "state": "def load(p):\n    return json.load(open(p))",
        "questions": {
          "bug":   {"type": "noul", "instructions": "Does this leak a file handle?"},
          "fix":   {"type": "choice", "criteria": {"with": "...", "close": "...", "none": "..."}},
          "grade": {"type": "score", "criteria": ["bad", "ok", "good"]}
        }
      }'
```

`questions` is a mapping of names to typed questions, answered independently of
their order. Types: `noul` (yes/no), `choice` (one option key), `score` (one
ordered level). The response carries, per question, the winning option and the
full probability distribution, so callers can threshold instead of taking an
argmax. `images` (base64 data URIs) and `videos` are oMLX extensions for the
backbones that read them.

### `mode`

Decider families (`d1`, `pplx_decider`) accept `mode`:

- `separate` (default) — one sequence per question. The state is read once per
  question; the prompt cache is reused across them.
- `isolated` — one sequence for the whole request. Cheaper on long states, at a
  small cost in answer agreement.

Non-causal attention means the state tokens see the question that follows them,
so a cached prefix cannot be shared between questions the way a causal decoder
can. On a long state, `isolated` trades a little agreement for one pass instead
of one pass per question.

## Quantization

Decision checkpoints survive `oQ` quantization: the readout weights, the
`*_config.json` files of the head and the head's own directories (such as the
Perplexity Decider `locate-span/` adapter) are copied as-is and stay in their
source precision, while the backbone quantizes like any other model. A checkpoint whose
`decision_config.json` declares `attention_mode: noncausal_full_attention` is
flagged so the block-sensitivity passes drop the causal masks, and text-only
output is rejected because the vision tower is part of the model.

## Tests

`tests/test_d1_lfm2.py` and `tests/test_pplx_decider.py` cover detection, config
validation, prompt construction, both modes and the answer shape without real
weights; the neighbouring `tests/test_d1_lfm2.py`, `tests/test_model_discovery.py`
and `tests/test_systemone.py` cover the shared plumbing.

## Limits

- The Perplexity Decider `locate` (span) adapter is not wired up yet; only
  `noul`, `choice` and `score` questions are served, and a `locate` question is
  rejected by the request schema. The release runs it in its own forward pass
  with an unmerged LoRA and a span head over the state tokens and a `<|locate|>`
  marker, returning `{start, end, text, score}` spans.
- `d1` on the LFM2-VL backbone is supported through the existing mlx-vlm
  `models.d1` class; LFM3-VL and LFM3-Omni backbones use the oMLX class.
