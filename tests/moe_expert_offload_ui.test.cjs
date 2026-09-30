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
    // The residency field offers the largest whole-expert residency that fits
    // the memory ceiling, as reported by /api/models, and applies it on click.
    const GB = 1024 ** 3;
    app.selectedModel = {id: 'moe', config_model_type: 'olmoe', moe_expert_offload_supported: true,
        moe_expert_offload_presets: [{fraction: .125, bytes: 2 * GB, fits: true}, {fraction: .75, bytes: 9 * GB, fits: false}],
        moe_expert_offload_fit_fraction: .375, moe_expert_offload_fit_bytes: 5 * GB};
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[enabled]:true, [fraction]:0.25});
    assert.equal(app.moeOffloadFitOption(), true);
    assert.equal(app.moeOffloadFitLabel(), 'modal.model_settings.moe_expert_offload_fit_label: 37.5% resident \u00b7 ~5.0 GB');
    assert.equal(app.moeOffloadNoFit(), false);
    app.applyMoeOffloadFit();
    assert.equal(app.modelSettings.moe_expert_offload_resident_percent, 37.5);
    assert.equal(app.modelSettings[fraction], 0.375);
    await app.saveModelSettings();
    assert.equal(payload[fraction], 0.375, 'the applied fit is saved as the resident fraction');
    app.selectedModel.moe_expert_offload_fit_fraction = .25;
    assert.equal(app.moeOffloadFitOption(), true);
    assert.equal(app.moeOffloadFitLabel(), 'modal.model_settings.moe_expert_offload_fit_label: 25% resident \u00b7 ~5.0 GB');
    app.selectedModel.moe_expert_offload_fit_fraction = null;
    assert.equal(app.moeOffloadFitOption(), false);
    assert.equal(app.moeOffloadNoFit(), true);
    app.selectedModel.moe_expert_offload_presets = [];
    assert.equal(app.moeOffloadNoFit(), false);
    assert.ok(offload.includes('x-show="moeOffloadFitOption()"'));
    assert.ok(offload.includes('@click="applyMoeOffloadFit()"'));
    assert.ok(offload.includes('x-show="moeOffloadNoFit()"'));
    for (const key of ['mtp_enabled', 'vlm_mtp_enabled', 'dflash_enabled']) {
        const scope = {modelSettings: {[enabled]:false, [key]:true}, selectedModel: {}};
        assert.equal(vm.runInNewContext(offload.match(/:disabled="([^"]+)"/)[1], scope), true);
        const lines = html.split('\n');
        const i = lines.findIndex(line => line.includes('@click=') && line.includes(`modelSettings.${key} = !modelSettings.${key}`));
        const disabled = lines[i+1].match(/:disabled="([^"]+)"/)[1];
        assert.equal(vm.runInNewContext(disabled, {modelSettings:{[enabled]:true}}), true);
    }
    const percent = 'moe_expert_offload_resident_percent';
    for (const [stored, displayed] of [[0.125, 12.5], [0.333, 33.3], [0.02, 2], [1, 100]]) {
        app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[enabled]:true, [fraction]:stored});
        assert.equal(app.modelSettings[percent], displayed);
        app.onMoeExpertOffloadResidentBlur();
        await app.saveModelSettings();
        assert.equal(payload[fraction], stored, 'untouched values must survive blur and save');
        assert.equal(app.modelSettings.moe_expert_offload_resident_touched, false);
    }
    for (const [input, saved] of [[5, 0.05], [10, 0.1], [12.5, 0.125], [33.3, 0.333], [95, 0.95]]) {
        app.modelSettings[percent] = input;
        app.onMoeExpertOffloadResidentPercent();
        assert.equal(app.moeExpertOffloadResidentInvalid(), false);
        app.onMoeExpertOffloadResidentBlur();
        await app.saveModelSettings();
        assert.equal(payload[fraction], saved);
        app.modelSettings = app.buildModelSettingsState(app.selectedModel, payload);
        assert.equal(app.modelSettings[percent], input);
    }
    for (const [input, settled] of [['', 5], [0, 5], [100, 95]]) {
        const previous = app.modelSettings[fraction];
        app.modelSettings[percent] = input;
        app.onMoeExpertOffloadResidentPercent();
        assert.equal(app.moeExpertOffloadResidentInvalid(), true);
        assert.equal(app.modelSettings[fraction], previous, 'partial input must not change the saved fraction');
        app.onMoeExpertOffloadResidentBlur();
        await app.saveModelSettings();
        assert.equal(app.modelSettings[percent], settled);
        assert.equal(payload[fraction], settled / 100);
    }
    assert.match(offload, /<input type="number" min="5" max="95" step="any" inputmode="decimal"/);
    assert.ok(offload.includes('@input="onMoeExpertOffloadResidentPercent()"'));
    assert.ok(offload.includes('@blur="onMoeExpertOffloadResidentBlur()"'));
    console.log('PASS: offload save/reopen, residency fit hint, spec toggle exclusion and fractional resident percentage');
})().catch(error => {console.error(error); process.exitCode = 1});
