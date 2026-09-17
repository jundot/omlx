"""Static regression contracts for chat generation controls and telemetry."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = ROOT / "omlx/admin/templates/chat.html"
I18N_DIR = ROOT / "omlx/admin/i18n"

I18N_KEYS = {
    "chat.resume",
    "chat.unload_model",
    "chat.discard_pause",
    "chat.pause_tooltip",
    "chat.force_output_tooltip",
    "chat.force_output_requesting",
    "chat.force_output_accepted",
    "chat.force_output",
    "chat.think_more",
    "chat.think_more_tooltip",
    "chat.think_more_tokens",
    "chat.think_more_tokens_hint",
    "chat.stats.now",
    "chat.stats.overall",
    "chat.stats.speculative_window",
    "chat.stats.speculative_efficiency",
    "chat.status.force_output_accepted",
    "chat.error.force_output_stale",
    "chat.error.force_output_unsupported",
    "chat.error.force_output_failed",
    "chat.error.stream_failed",
    "chat.error.pause_failed",
    "chat.status.paused_safe",
    "chat.status.paused_unloaded",
    "chat.status.resuming",
    "chat.status.unloading_model",
    "chat.status.unload_unconfirmed",
    "chat.error.unload_paused_failed",
    "chat.error.unload_status_missing",
    "chat.error.unload_timeout",
    "chat.error.unload_confirmation_interrupted",
    "chat.error.checkpoint_missing",
    "chat.error.checkpoint_legacy",
    "chat.error.resume_failed",
}


def _source() -> str:
    return CHAT_TEMPLATE.read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_live_and_recorded_stats_include_recent_and_overall_metrics():
    source = _source()
    fields = {
        "generation_tokens_per_second_recent",
        "speculative_decoding_efficiency",
        "speculative_decoding_efficiency_recent",
        "speculative_accepted_tokens",
        "speculative_proposed_tokens",
        "speculative_efficiency_kind",
    }

    for field in fields:
        assert source.count(field) >= 4, field
    assert "formatRate(recentStats?.generation_tokens_per_second_recent)" in source
    assert "formatRate(recentStats?.avg_generation_tps)" in source
    assert (
        "formatEfficiency(recentStats?.speculative_decoding_efficiency_recent)"
        in source
    )
    assert (
        "formatEfficiency(recentStats?.speculative_decoding_efficiency)" in source
    )
    assert "if (value == null || value === '') return '\\u2014';" in source


def test_answer_now_uses_authoritative_sse_request_id():
    source = _source()
    force_output = _section(
        source, "async forceCurrentStreamOutput()", "pausedPreviewFromStream(stream)"
    )

    assert "stream?.thinkingState?.isInThinking" in force_output
    assert "'/v1/requests/'" in force_output
    assert "'/force-output'" in force_output
    assert "'Authorization': `Bearer ${this.getApiKey()}`" in force_output
    assert "response.status === 202" in force_output
    assert "response.status === 409" in force_output
    assert "response.status === 404" in force_output
    assert "response.status === 422" in force_output
    assert "stream.requestId = data.id" in source
    assert "'X-Request-ID': stream.requestId" in source


def test_pause_checkpoint_is_durable_before_transport_cancellation():
    source = _source()
    pause = _section(source, "async pauseStreaming()", "async finalizePausedStream")
    can_pause = _section(
        source, "canPauseCurrentStream()", "async forceCurrentStreamOutput()"
    )

    first_write = pause.index("await putPausedInferenceCheckpoint(checkpoint)")
    cancel = pause.index("this.cancelStreamTransport(stream)")
    assert first_write < cancel
    assert "tx.oncomplete = () => resolve(result)" in source
    assert pause.count("await putPausedInferenceCheckpoint(checkpoint)") == 1
    assert "'/v1/requests/'" in pause
    assert "'/pause'" in pause
    assert "snapshot.output_token_ids" in pause
    assert "snapshot.reasoning_end_token_index" in pause
    assert "snapshot.continuation_in_thinking" in pause
    assert "version: 2" in pause
    assert "baseRequestBody: this.cloneData(stream.baseRequestBody)" in pause
    assert "outputTokenIds:" in pause
    assert "targetMessageId: stream.targetMessageId" in pause
    assert "settings: this.captureSessionModelSettings()" in pause
    assert "toolRoundContext:" in pause
    checkpoint_object = pause.split("const checkpoint = {", 1)[1].split("};", 1)[0]
    assert "getApiKey" not in checkpoint_object
    assert "!stream.isToolExecuting" in can_pause


def test_cancelled_reader_done_path_finalizes_pause_instead_of_completion():
    source = _source()
    stream = _section(
        source, "async streamResponse(streamContext = null, depth = 0)", "stopStreaming()"
    )

    loop_end = stream.index("if (done) break")
    pause_done_path = stream.index("if (stream.pauseRequested)", loop_end)
    normal_completion = stream.index(
        "// Always push the completed assistant message", pause_done_path
    )
    assert pause_done_path < normal_completion
    assert "await this.finalizePausedStream(context, stream, chatSession)" in stream[
        pause_done_path:normal_completion
    ]
    assert "throw new DOMException('Stream aborted', 'AbortError')" in stream[
        pause_done_path:normal_completion
    ]
    catch = stream.split("console.log('[streamResponse] catch error:'", 1)[1]
    assert catch.index("if (stream.pauseRequested)") < catch.index(
        "if (error.name === 'AbortError')"
    )
    assert "let pauseFinalized = false" in stream
    assert "!thinkMoreRestart && !pauseFinalized" in stream
    assert "if (depth === 0 && pauseFinalized)" in stream


def test_finalized_checkpoint_is_unwrapped_from_alpine_before_indexeddb_write():
    source = _source()
    finalize = _section(
        source, "async finalizePausedStream", "pausedActionBusy(message)"
    )

    assert "const persistedCheckpoint = this.cloneData(checkpoint)" in finalize
    assert "await putPausedInferenceCheckpoint(persistedCheckpoint)" in finalize
    assert "this.buildPausedMessage(persistedCheckpoint)" in finalize


def test_resume_continues_from_exact_token_prefix_and_keeps_message_identity():
    source = _source()
    resume = _section(
        source, "async resumePausedMessage(message)", "async restorePausedCheckpoints()"
    )
    stream = _section(
        source, "async streamResponse(streamContext = null, depth = 0)", "stopStreaming()"
    )

    assert "checkpoint.version !== 2" in resume
    assert "chat.error.checkpoint_legacy" in resume
    assert "this.buildResumeRequestBody(" in resume
    assert "checkpoint.baseRequestBody" in resume
    assert "checkpoint.outputTokenIds" in resume
    assert "checkpoint.continuationInThinking" in resume
    assert "_requestBodyOverride: resumeBody" in resume
    assert "_initialOutputTokenIds: checkpoint.outputTokenIds" in resume
    assert "_initialPreview: checkpoint.preview" in resume
    assert "_targetMessageId: checkpoint.targetMessageId" in resume
    assert "const previousState = paused.state" in resume
    assert "previousState === 'unloaded' ? 'unloaded' : 'paused'" in resume
    assert "stream.baseRequestBody = context._baseRequestBody" in stream
    assert "stream.outputTokenIds = this.normalizeOutputTokenIds(" in stream
    assert "delta?.token_ids" in stream
    assert "delta?.reasoning_token_count" in stream
    assert "stream.targetMessageId = context._targetMessageId" in stream
    assert "context._resumeCheckpointId && context._resumeSucceeded" in stream
    assert "await deletePausedInferenceCheckpoint(context._resumeCheckpointId)" in stream
    assert "stream.pauseRequested" in stream
    assert "await this.finalizePausedStream(context, stream, chatSession)" in stream


def test_resume_preserves_original_total_generation_and_reasoning_caps():
    source = _source()
    builder = _section(
        source, "buildResumeRequestBody(baseRequestBody", "previewFromPausedSnapshot"
    )

    assert "Math.floor(originalMax) - ids.length" in builder
    assert "Math.floor(originalThinkingBudget)" in builder
    assert "Number(reasoningTokenCount)" in builder
    assert "Math.max(" in builder


def test_prefill_pause_resumes_original_request_without_empty_continuation():
    source = _source()
    builder = _section(
        source,
        "buildContinuationRequestBody(baseRequestBody",
        "buildResumeRequestBody(baseRequestBody",
    )

    assert "if (ids.length)" in builder
    assert "body.continuation_token_ids = ids" in builder
    assert builder.index("if (ids.length)") < builder.index(
        "body.continuation_token_ids = ids"
    )


def test_chat_requests_opt_into_raw_output_token_ids():
    source = _source()
    assert "stream_options: { include_usage: true, include_token_ids: true }" in source
    strip = _section(source, "stripContinuationFields(requestBody)", "buildContinuationRequestBody")
    assert "include_token_ids: true" in strip


def test_think_more_discards_answer_and_continues_only_reasoning_prefix():
    source = _source()
    controls = _section(source, "canThinkMoreCurrentStream()", "async regenerateMessage")

    assert "reasoningPrefixTokenIds" in controls
    assert "ids.slice(0, count)" in source
    assert "_initialContinuationInThinking: true" in controls
    assert "_initialPreview:" in controls
    assert "content: ''" in controls
    assert "thinking_budget: extra" in source
    assert "Math.floor(originalMax) - retainedTokenCount" in source
    assert "generationOverride.thinking_budget + reasoningIds.length" in controls
    assert "_replaceMessageId = message.id" in controls
    assert "_replacementPreviewMessage = this.cloneData(message)" in controls
    assert "stream.thinkMoreRestart = restart" in controls
    assert "if (stream.thinkMoreRestart)" in source
    assert "if (depth === 0 && !thinkMoreRestart && !pauseFinalized)" in source
    assert "await this.streamResponse(thinkMoreRestart, 0)" in source


def test_think_more_completion_upserts_stable_message_id_without_duplicates():
    source = _source()
    helper = _section(source, "upsertMessageById(messages", "resolveGatewayModelId")
    stream = _section(source, "async streamResponse", "stopStreaming()")

    assert "[message.id, replaceMessageId].filter(Boolean)" in helper
    assert "messages.splice(i, 1)" in helper
    assert "Object.assign(existing, message)" in helper
    assert "messages.splice(insertIndex, 1, message)" in helper
    assert "if (i === insertIndex) continue" in helper
    assert (
        "this.upsertMessageById(\n"
        "                chatSession.messages, assistantMsg, context._replaceMessageId\n"
        "            )"
    ) in stream
    assert "this.upsertMessageById(chatSession.messages, assistantMsg)" in stream
    assert stream.count(
        "chatSession.messages, errorMsg, context._replaceMessageId"
    ) >= 2
    controls = _section(source, "thinkMoreCurrentStream()", "async thinkMoreMessage")
    assert "restart._replaceMessageId = null" not in controls
    assert "restart._replacementPreviewMessage = null" not in controls


def test_think_more_budget_is_configurable_and_persisted_with_chat_settings():
    source = _source()

    assert source.count("think_more_budget_tokens: 4096") >= 2
    assert 'x-model.number="modelSettings.think_more_budget_tokens"' in source
    assert 'min="1" step="1024"' in source
    capture = _section(source, "captureSessionModelSettings()", "saveModelSettingsForModel")
    assert "normalizeThinkMoreBudgetTokens" in capture
    assert "modelSettings: this.cloneData(this.modelSettings)" in capture
    assert 'x-show="canThinkMoreMessage(msg, index)"' in source
    assert 'x-show="canThinkMoreCurrentStream()"' in source


def test_paused_checkpoints_restore_after_reload_and_offer_all_actions():
    source = _source()
    paused_actions = _section(
        source, '<button @click="resumePausedMessage(msg)"', "</template>"
    )

    assert "const PAUSED_INFERENCE_DB = 'omlx-paused-inference'" in source
    assert "await this.restorePausedCheckpoints()" in source
    assert "resumePausedMessage(msg)" in source
    assert ':disabled="pausedActionBusy(msg)"' in paused_actions
    resume_button = paused_actions.split("</button>", 1)[0]
    assert "state === 'unloaded'" not in resume_button
    assert "unloadPausedModel(msg)" in source
    assert "discardPausedCheckpoint(msg)" in source
    assert "'/unload'" in source


def test_open_reasoning_pause_does_not_coerce_null_boundary_to_zero():
    source = _source()
    pause = _section(source, "async pauseStreaming()", "async finalizePausedStream")

    assert "const rawReasoningEnd = snapshot.reasoning_end_token_index" in pause
    assert "rawReasoningEnd != null" in pause
    assert "? checkpoint.outputTokenIds.length : 0" in pause


def test_queued_unload_stays_busy_until_model_status_confirms_completion():
    source = _source()
    unload = _section(
        source, "async unloadPausedModel(message)", "async waitForPausedModelUnload"
    )
    poll = _section(
        source, "async waitForPausedModelUnload", "async resumePausedMessage(message)"
    )
    paused_actions = _section(
        source, '<button @click="resumePausedMessage(msg)"', "</template>"
    )

    assert "paused.state = 'unloading'" in unload
    assert "await this.waitForPausedModelUnload(paused.modelId)" in unload
    assert unload.index("await this.waitForPausedModelUnload") < unload.index(
        "paused.state = 'unloaded'"
    )
    assert "fetchAdminModelsList({ force: true })" in poll
    assert "!model.loaded && !model.is_loading" in poll
    assert "timeoutMs = 30000" in poll
    assert "chat.error.unload_timeout" in poll
    assert "paused.state = 'unload_error'" in unload
    resume_button = paused_actions.split("</button>", 1)[0]
    assert "state === 'unload_error'" in resume_button
    assert "state === 'unloaded'" not in resume_button
    assert "paused.state === 'unload_error'" in source
    assert "['unloaded', 'unload_error'].includes(paused.state)" in source
    assert "chat.error.unload_confirmation_interrupted" in source


def test_generation_control_i18n_keys_exist_in_every_locale():
    for locale_path in sorted(I18N_DIR.glob("*.json")):
        translations = json.loads(locale_path.read_text(encoding="utf-8"))
        missing = I18N_KEYS - translations.keys()
        assert not missing, f"{locale_path.name} is missing {sorted(missing)}"
