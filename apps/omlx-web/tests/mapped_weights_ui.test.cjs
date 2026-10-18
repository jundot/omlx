// SPDX-License-Identifier: Apache-2.0
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert/strict');
const root = path.resolve(__dirname, '..');
const context = {localStorage: {getItem: () => null}, window: {t: k => k}, console,
    alert: message => {throw Error(message)}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'omlx_web/static/js/dashboard.js'), 'utf8'), context);
(async () => {
    const app = context.dashboard();
    app.loadModels = async () => {};
    app.selectedModel = {id: 'ornith', config_model_type: 'qwen3_5_moe', moe_expert_offload_supported: true};
    let payload;
    context.fetch = async (_, init) => {
        payload = JSON.parse(init.body);
        return {ok: true, json: async () => ({})};
    };
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {mapped_weights_enabled: false});
    app.modelSettings.mapped_weights_enabled = true;
    await app.saveModelSettings();
    assert.equal(payload.mapped_weights_enabled, true);
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, payload);
    assert.equal(app.modelSettings.mapped_weights_enabled, true);
    const html = fs.readFileSync(path.join(root, 'omlx_web/templates/dashboard/_modal_model_settings.html'), 'utf8');
    const mapped = html.split('<!-- Mapped weights -->')[1].split('<!-- MoE Expert Offload -->')[0];
    const disabled = mapped.match(/:disabled="([^"]+)"/)[1];
    assert.equal(vm.runInNewContext(disabled, {modelSettings:{moe_expert_offload_enabled:true}}), true);
    const offload = html.split('<!-- MoE Expert Offload -->')[1].split('<!-- TurboQuant KV Cache -->')[0];
    assert.equal(vm.runInNewContext(offload.match(/:disabled="([^"]+)"/)[1],
        {selectedModel:null, modelSettings:{mapped_weights_enabled:true}}), true);
    const texts = {ok: '{mappable}/{total} {percent}', low: 'low {percent}'};
    assert.equal(app.mappedWeightsNote(null, texts), null);
    assert.equal(app.mappedWeightsNote({mappable_bytes: 0, total_bytes: 0}, texts), null);
    const aligned = app.mappedWeightsNote({mappable_bytes: 9 * 2 ** 30, total_bytes: 10 * 2 ** 30}, texts);
    assert.equal(aligned.low, false);
    assert.equal(aligned.text, '9.0 GB/10.0 GB 90');
    const misaligned = app.mappedWeightsNote({mappable_bytes: 2 ** 30, total_bytes: 16 * 2 ** 30}, texts);
    assert.equal(misaligned.low, true);
    assert.equal(misaligned.text, 'low 6');
    console.log('PASS: mapped weights save/reopen, expert offload exclusion and alignment note');
})().catch(error => {console.error(error); process.exitCode = 1});
