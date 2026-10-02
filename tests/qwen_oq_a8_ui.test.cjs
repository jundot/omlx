const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(
    path.join(__dirname, '../omlx/admin/static/js/dashboard.js'), 'utf8'
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
]) {
    assert.equal(state.isQwenOqA8Model({config_model_type}), true);
}
for (const config_model_type of [
    '', 'qwen2', 'qwen3', 'qwen4_exp', 'llama', 'gemma4', 'k2_horizon',
]) {
    assert.equal(state.isQwenOqA8Model({config_model_type}), false);
}
assert.equal(state.isQwenOqA8Model(null), false);

state.modelSettings.qwen35_oq_a8_enabled = true;
state.modelSettings.qwen35_ane_prefill_enabled = true;
assert.equal(state.validateQwenOqA8Settings(), 'js.error.ane_oq_a8_conflict');
state.modelSettings.qwen35_ane_prefill_enabled = false;
assert.equal(state.validateQwenOqA8Settings(), null);
console.log('Architecture-only visibility and ANE conflict checks passed');

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
})().catch(error => {console.error(error); process.exitCode = 1});
