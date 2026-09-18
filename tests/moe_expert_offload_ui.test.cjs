// SPDX-License-Identifier: Apache-2.0
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const root = path.resolve(__dirname, '..');
const context = {localStorage: {getItem: () => null}, window: {t: k => k}, console,
    alert: message => {throw Error(message)}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'omlx/admin/static/js/dashboard.js'), 'utf8'), context);
(async () => {
    const app = context.dashboard();
    app.loadModels = async () => {};
    app.selectedModel = {id: 'v41', config_model_type: 'deepseek_v41', moe_expert_offload_supported: true};
    let payload;
    context.fetch = async (_, init) => {
        payload = JSON.parse(init.body);
        return {ok: true, json: async () => ({})};
    };
    const enabled = 'moe_expert_offload_enabled', fraction = 'moe_expert_offload_resident_fraction';
    for (const value of [.125, .25, .5, .75]) {
        app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[enabled]:true, [fraction]:value});
        await app.saveModelSettings();
        assert.equal(payload[enabled], true);
        assert.equal(payload[fraction], value);
        app.modelSettings = app.buildModelSettingsState(app.selectedModel, payload);
        assert.equal(app.modelSettings[fraction], value);
    }
    const html = fs.readFileSync(path.join(root, 'omlx/admin/templates/dashboard/_modal_model_settings.html'), 'utf8');
    const offload = html.split('<!-- MoE Expert Offload -->')[1].split('<!-- IndexCache')[0];
    const condition = offload.match(/x-if="([^"]+)"/)[1];
    for (const supported of [true, false, undefined]) {
        assert.equal(vm.runInNewContext(condition, {selectedModel:{moe_expert_offload_supported:supported}}), supported === true);
    }
    // Residency options carry the admission size per option, mark the ones
    // over the ceiling, and offer the fit when it is not one of the presets.
    const GB = 1024 ** 3;
    app.selectedModel = {id: 'moe', config_model_type: 'olmoe', moe_expert_offload_supported: true,
        moe_expert_offload_presets: [{fraction: .125, bytes: 2 * GB, fits: true}, {fraction: .75, bytes: 9 * GB, fits: false}],
        moe_expert_offload_fit_fraction: .375, moe_expert_offload_fit_bytes: 5 * GB};
    assert.equal(app.moeOffloadPresetLabel(.125, 'modal.model_settings.moe_expert_offload_resident_12_5'), 'modal.model_settings.moe_expert_offload_resident_12_5 \u00b7 ~2.0 GB');
    assert.equal(app.moeOffloadPresetLabel(.75, 'modal.model_settings.moe_expert_offload_resident_75'), 'modal.model_settings.moe_expert_offload_resident_75 \u00b7 ~9.0 GB \u00b7 modal.model_settings.moe_expert_offload_exceeds');
    assert.equal(app.moeOffloadPresetLabel(.5, 'modal.model_settings.moe_expert_offload_resident_50'), 'modal.model_settings.moe_expert_offload_resident_50');
    assert.equal(app.moeOffloadFitOption(), true);
    assert.equal(app.moeOffloadFitLabel(), 'modal.model_settings.moe_expert_offload_fit_label: 37.5% resident \u00b7 ~5.0 GB');
    assert.equal(app.moeOffloadNoFit(), false);
    app.selectedModel.moe_expert_offload_fit_fraction = .25;
    assert.equal(app.moeOffloadFitOption(), false);
    assert.equal(app.moeOffloadFitLabel(), 'modal.model_settings.moe_expert_offload_fit_label: 25% resident \u00b7 ~5.0 GB');
    app.selectedModel.moe_expert_offload_fit_fraction = null;
    assert.equal(app.moeOffloadFitOption(), false);
    assert.equal(app.moeOffloadNoFit(), true);
    app.selectedModel.moe_expert_offload_presets = [];
    assert.equal(app.moeOffloadNoFit(), false);
    delete app.selectedModel.moe_expert_offload_presets;
    assert.equal(app.moeOffloadPresetLabel(.125, 'modal.model_settings.moe_expert_offload_resident_12_5'), 'modal.model_settings.moe_expert_offload_resident_12_5');
    assert.ok(offload.includes('x-if="moeOffloadFitOption()"'));
    assert.ok(offload.includes(':value="selectedModel.moe_expert_offload_fit_fraction"'));
    for (const value of [0.125, 0.25, 0.5, 0.75]) assert.ok(offload.includes(`<option value="${value}"`));
    for (const key of ['mtp_enabled', 'vlm_mtp_enabled', 'dflash_enabled']) {
        const scope = {modelSettings: {[enabled]:false, [key]:true}};
        assert.equal(vm.runInNewContext(offload.match(/:disabled="([^"]+)"/)[1], scope), true);
        const lines = html.split('\n');
        const i = lines.findIndex(line => line.includes('@click=') && line.includes(`modelSettings.${key} = !modelSettings.${key}`));
        const disabled = lines[i+1].match(/:disabled="([^"]+)"/)[1];
        assert.equal(vm.runInNewContext(disabled, {modelSettings:{[enabled]:true}}), true);
    }
    console.log('PASS: offload save/reopen, residency sizing labels, and bidirectional speculative toggle exclusion');
})().catch(error => {console.error(error); process.exitCode = 1});
