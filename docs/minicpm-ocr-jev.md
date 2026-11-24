# MiniCPM-OCR-Jev

See the [official OpenBMB model card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)
and [training prompt and input contract](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/finetune_jev_en.md#dataset-format).

oMLX serves MiniCPM-OCR-Jev through `POST /v1/decisions` and
`POST /v1/systemone`. It verifies OCR region consistency with a binary classifier.
The candidate OCR text is preserved exactly, including whitespace and literal
prefixes. Every request supplies exactly two images in order: **[original crop, rendered
candidate text]**, plus the candidate OCR text. Render the candidate yourself;
oMLX does not generate the second image. Extreme crops are padded with white to
at most 199:1 before image preprocessing.

The classifier uses its fixed training prompt. Question instructions and option
or level descriptions do not change what it judges. Multiple questions are
projections of the same binary result, rather than separate judgments:

- `predicate` returns the probability that the candidate text and rendering match
  the original crop.
- `choice` requires two recognizable positive/negative values, such as
  `"correct"`/`"incorrect"`, `"match"`/`"mismatch"`, or `true`/`false`.
- `score` requires exactly two levels: index 0 means mismatch and index 1 means
  match. Display labels may be arbitrary. The returned score is the expected
  level between 0 and 1, numerically equal to the match probability.
  Choice/score confidence is the absolute difference between the two class
  probabilities.

The model is an OCR verifier; arbitrary classification tasks and videos are
unsupported. Inputs exceeding the model context return an error rather than
truncating the OCR prompt.

## Checkpoint identity

Checkpoints declaring `model_type: "minicpm_ocr_jev"` are discovered directly.
The **official OpenBMB release also needs the identity sidecar below**: its
Qwen3.5 config declares a classifier without an OCR-specific model type.
For that release and unmarked conversions, place this `decision_config.json`
beside `config.json` after verifying that the checkpoint uses the official
binary OCR head and prompt:

```json
{
  "format_id": "minicpm-ocr-jev-v1",
  "prompt_template_md5": "9aab68e8ba8ab4da17ba9752bddb199c",
  "images_per_prompt": 2
}
```

A generic Qwen binary classifier is insufficient to identify this OCR contract.
The sidecar declares the adapter identity without changing weights. oMLX oQ
conversion preserves it; when using other conversion tools, copy it explicitly
and check model discovery again. Keep the vision tower and binary `score` head.
New oQ conversions keep the small binary head floating point; existing exports
with a quantized head continue to load.

## Request

For exact candidate fidelity on `/v1/decisions`, send exactly one `input_text`
part containing the raw candidate, including an explicit empty string for an
empty candidate. MiniCPM rejects missing or multiple text parts. Direct
`/v1/systemone` raw string input is preserved exactly.

This Python example reads two PNGs, sends valid data URLs, and prints the match
probability. Set `OMLX_API_KEY` if your server requires authentication.

```python
import base64
import os
from pathlib import Path

import requests


def png_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


response = requests.post(
    "http://localhost:8000/v1/decisions",
    headers={"Authorization": f"Bearer {os.environ.get('OMLX_API_KEY', '')}"},
    json={
        "model": "MiniCPM-OCR-Jev-oQ8",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": png_url("original.png")},
                {"type": "input_image", "image_url": png_url("render.png")},
                {"type": "input_text", "text": "Invoice total: $123.45"},
            ],
        }],
        "questions": [{
            "type": "predicate",
            "name": "match",
            "instructions": "Does the OCR match the original?",
        }],
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["answers"][0]["probability"])
```

For `/v1/systemone`, use `state` for the candidate string (or
`{"ocr_text": "..."}`), `images` for the same two ordered data URLs, and
`questions: {"match": {"type": "noul"}}`. The result is
`answers.match.noul`, the same match probability. Scores and choices use the
shared System One schema.

## Optional checkpoint parity check

Default tests use tiny local tensors and do not download weights. To run the
local checkpoint parity test, set `OMLX_MINICPM_JEV_MODEL` to a checkpoint folder
and `OMLX_MINICPM_JEV_EXAMPLES` to a fixture folder, then run:

```bash
python -m pytest tests/test_decision_minicpm_jev.py -m slow -k reference_parity
```

The fixture folder is user-prepared; it is not shipped or downloaded by the
test. Prepare reference scores with the official model and matching prompt. It contains `01_text_putnam`, `02_formula_putnam`, and
`03_table_gb_standard`. Each has a `case.json` with `reference_render_of`, a
`candidates` list of `{role, text}`, `observed_prob_yes` keyed by role,
`block_image`, and `reference_render` filenames. The corresponding image files
must be present. This integration test is skipped unless both paths are set.
