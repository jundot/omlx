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
    window: {},
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
assert.match(state.validateQwenOqA8Settings(), /cannot both/);
state.modelSettings.qwen35_ane_prefill_enabled = false;
assert.equal(state.validateQwenOqA8Settings(), null);
console.log('Architecture-only visibility and ANE conflict checks passed');
