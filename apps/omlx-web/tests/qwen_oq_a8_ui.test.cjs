const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(
    path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8'
);
const context = {
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme',
    ENHANCED_READABILITY_KEY: 'readability',
    window: {t: key => key},
    navigator: {language: 'en'},
    document: {},
};
const create = vm.runInNewContext(source + '\n dashboard;', context);
const state = create();

for (const config_model_type of [
    'qwen3_5', 'qwen3_5_moe', 'qwen3_6', 'qwen3_8', 'Qwen3-8',
    'qwen4_exp', 'Qwen4-Exp', 'deepseek_v41',
]) {
    assert.equal(state.isA8PrefillModel({config_model_type}), true);
}
for (const config_model_type of [
    '', 'qwen2', 'qwen3', 'qwen4', 'qwen4_exp_x', 'llama', 'gemma4', 'k2_horizon',
    'deepseek_v4',
]) {
    assert.equal(state.isA8PrefillModel({config_model_type}), false);
}
assert.equal(state.isA8PrefillModel(null), false);

// Hidden where the kernels cannot run, unless already on.
assert.equal(state.showsA8Prefill({config_model_type: 'qwen3_5', a8_prefill_supported: false}), false);
assert.equal(state.showsA8Prefill({config_model_type: 'qwen3_5', a8_prefill_supported: true}), true);
assert.equal(state.showsA8Prefill({
    config_model_type: 'qwen3_5',
    a8_prefill_supported: false,
    settings: {a8_prefill_enabled: true},
}), true);
assert.equal(state.showsA8Prefill({config_model_type: 'llama', a8_prefill_supported: true}), false);

state.modelSettings.a8_prefill_enabled = true;
state.modelSettings.qwen35_ane_prefill_enabled = true;
assert.equal(state.validateA8PrefillSettings(), 'js.error.ane_oq_a8_conflict');
state.modelSettings.qwen35_ane_prefill_enabled = false;
assert.equal(state.validateA8PrefillSettings(), null);
console.log('oQ A8 visibility and ANE conflict checks passed');

(async () => {
    const app = create();
    app.loadModels = async () => {};
    app.selectedModel = {id: 'qwen', config_model_type: 'qwen3_5'};
    let payload;
    context.fetch = async (_url, init) => {
        payload = JSON.parse(init.body);
        return {ok: true, json: async () => ({})};
    };
    context.alert = message => { throw Error(message); };
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {mtp_enabled: true, mtp_fixed_depth: 2});
    assert.equal(app.modelSettings.mtp_adaptive_max_depth, '3');
    for (const depth of [3, 4, 5, 6]) {
        app.modelSettings.mtp_adaptive_max_depth = String(depth);
        await app.saveModelSettings();
        assert.equal(payload.mtp_adaptive_max_depth, depth);
        assert.equal(payload.mtp_fixed_depth, null);
        const restored = app.buildModelSettingsState(app.selectedModel, payload);
        assert.equal(restored.mtp_adaptive_max_depth, String(depth));
    }
    console.log('Lightning MTP adaptive maximum round-trip and fixed-depth reset passed');
})().then(async () => {
    // Qwen3.8-Flash-Next (qwen4_exp): the same toggle and floor are saved and restored.
    const app = create();
    app.loadModels = async () => {};
    app.selectedModel = {id: 'flash', config_model_type: 'qwen4_exp'};
    assert.equal(app.isA8PrefillModel(app.selectedModel), true);
    let payload;
    context.fetch = async (_url, init) => {
        payload = JSON.parse(init.body);
        return {ok: true, json: async () => ({})};
    };
    context.alert = message => { throw Error(message); };
    app.modelSettings = app.buildModelSettingsState(
        app.selectedModel, {a8_prefill_enabled: true, a8_prefill_min_tokens: 256});
    assert.equal(app.modelSettings.a8_prefill_enabled, true);
    assert.equal(app.modelSettings.a8_prefill_min_tokens, 256);
    await app.saveModelSettings();
    assert.equal(payload.a8_prefill_enabled, true);
    assert.equal(payload.a8_prefill_min_tokens, 256);
    const restored = app.buildModelSettingsState(app.selectedModel, payload);
    assert.equal(restored.a8_prefill_enabled, true);
    assert.equal(restored.a8_prefill_min_tokens, 256);
    console.log('qwen4_exp oQ A8 toggle and minimum-token round-trip passed');
}).catch(error => {console.error(error); process.exitCode = 1});
