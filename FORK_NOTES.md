# Fork Issue Triage Notes

Running log of non-obvious findings while working through upstream (`jundot/omlx`)
issues in this fork. One doc, appended over time.

---

## #2332 — tool-call fallback parsers drop malformed array/object params

**Status in this fork: core already fixed; one flagged gap remains.**

- The headline bug (the three XML/namespaced fallback parsers keeping a
  malformed array/object param as a raw string) is **already fixed** here by
  commit `093e7df1` — *"fix: schema-aware param coercion in XML tool call
  fallbacks (#2332)"*. It threads the tool schema into
  `_parse_xml_tool_calls()` and `_parse_namespaced_tool_calls()` and adds a
  bracket-repair pass via `_coerce_param_value()` / `_repair_json_value()`.
  Verified: the reporter's exact repro now returns a proper `list` (emits a
  `Repaired malformed JSON for parameter 'edits'...` log line). Reporter also
  confirmed the fix on the issue thread.

- **Remaining gap (the "fourth spot" the issue explicitly flagged):**
  `_parse_bracket_tool_calls()` (`omlx/api/tool_calling.py:531`) was *not*
  touched by that commit (scoped to "XML tool call fallbacks"). It is called
  without `tools` (`:1421`), and on a malformed args object it does
  `arguments = {"raw": args_str}` — which drops **all** real parameters and
  replaces them with a bogus `raw` key (arguably worse than the original bug).
  Reproduced: `[Calling tool: edit({...malformed...})]` → `{"raw": "..."}`.
  A clean fix would thread `tools` through and reuse the existing
  `_coerce_param_value` / `_repair_json_value` helpers. Lower real-world
  frequency than the XML format (reporter's 19/19 live occurrences were all the
  `<function=name>` XML format), so deprioritized for now.

**Env note:** the project's Python env is `./.venv` (has pydantic, mlx, etc.).
System `python3` lacks deps — always run repros/tests via `./.venv/bin/python`.

---

## #2367 — `enable_thinking` has no effect on Qwen3 via OpenAI API

**Status: fixed here (top-level toggle now honored).**

- Ground truth (matches maintainer thread): the **nested** form
  `chat_template_kwargs: {"enable_thinking": false}` already works — it flows
  through `merge_chat_template_request_kwargs()` (`omlx/model_settings.py:1214`)
  to the render, honoring the `forced_ct_kwargs` "force lock". The
  "reasoning-in-content" rows in the report are the max_tokens truncation
  fallback (generation cut before `</think>`, `finish_reason: "length"`) — that
  fallback is intentional.
- Real gap: a **top-level** `enable_thinking` (the name used by the Qwen model
  card and vMLX) was *silently dropped* — `ChatCompletionRequest` had no such
  field and pydantic defaults to `extra="ignore"`. Accept-then-ignore is the
  footgun the reporter hit.
- Fix (`omlx/api/openai_models.py`): added `enable_thinking: Optional[bool]` to
  `ChatCompletionRequest` plus a `model_validator(mode="after")` that folds it
  into `chat_template_kwargs["enable_thinking"]` (nested value wins;
  `setdefault`). Doing it at the schema layer means every server path
  (streaming + non-streaming) picks it up and the force-lock stays authoritative.
  Tests in `tests/test_thinking_toggle.py::TestRequestEnableThinkingToggle`.
- **Not verified end-to-end** against a live model (Qwen3.6-35B-A3B needs a
  multi-GB MLX load); the schema→merge runtime path is covered by unit tests.
  Did **not** add a `thinking` bool alias (collides with Anthropic's `thinking`
  object semantics).
