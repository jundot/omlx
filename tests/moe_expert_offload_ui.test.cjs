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
    for (const key of ['mtp_enabled', 'vlm_mtp_enabled', 'dflash_enabled']) {
        const scope = {modelSettings: {[enabled]:false, [key]:true}};
        assert.equal(vm.runInNewContext(offload.match(/:disabled="([^"]+)"/)[1], scope), true);
        const lines = html.split('\n');
        const i = lines.findIndex(line => line.includes('@click=') && line.includes(`modelSettings.${key} = !modelSettings.${key}`));
        const disabled = lines[i+1].match(/:disabled="([^"]+)"/)[1];
        assert.equal(vm.runInNewContext(disabled, {modelSettings:{[enabled]:true}}), true);
    }
    const percent = 'moe_expert_offload_resident_percent';
    // A stored fraction reopens as its whole percentage.
    for (const value of [0.33, 0.4, 0.8]) {
        app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[enabled]:true, [fraction]:value});
        assert.equal(app.modelSettings[percent], Math.round(value * 100));
    }
    // An out-of-range stored value (the old 12.5% preset) opens silently: the
    // field reports an error only once the user has typed in it.
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[enabled]:true, [fraction]:0.125});
    assert.equal(app.modelSettings[percent], 13);
    assert.ok(app.moeExpertOffloadResidentInvalid());
    assert.equal(app.modelSettings.moe_expert_offload_resident_touched, false);
    app.onMoeExpertOffloadResidentPercent();
    assert.equal(app.modelSettings.moe_expert_offload_resident_touched, true);
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[enabled]:true, [fraction]:0.33});
    await app.saveModelSettings();
    assert.equal(payload[fraction], 0.33);
    // Editing the percentage moves the fraction that gets saved.
    app.modelSettings[percent] = 40;
    app.onMoeExpertOffloadResidentPercent();
    await app.saveModelSettings();
    assert.equal(payload[fraction], 0.4);
    // Only two-digit percentages land: the "0" on the way to "50", a stray low
    // or high value, and a third digit all leave the fraction alone.
    for (const inert of [0, 15, 85, 100]) {
        app.modelSettings[percent] = inert;
        app.onMoeExpertOffloadResidentPercent();
        assert.equal(app.modelSettings[fraction], 0.4, `${inert} must not rewrite the fraction`);
    }
    app.modelSettings[percent] = 50;
    app.onMoeExpertOffloadResidentPercent();
    assert.equal(app.modelSettings[fraction], 0.5);
    // Out-of-range input reports itself rather than silently doing nothing.
    for (const [value, invalid] of [[15, true], [20, false], [80, false], [85, true]]) {
        app.modelSettings[percent] = value;
        assert.equal(app.moeExpertOffloadResidentInvalid(), invalid, `${value}`);
    }
    // Settling the field brings it back into range instead of leaving a number
    // that could never be saved.
    app.modelSettings[percent] = 85;
    app.onMoeExpertOffloadResidentBlur();
    assert.equal(app.modelSettings[percent], 80);
    assert.equal(app.modelSettings[fraction], 0.8);
    app.modelSettings[percent] = 15;
    app.onMoeExpertOffloadResidentBlur();
    assert.equal(app.modelSettings[percent], 20);
    assert.equal(app.modelSettings[fraction], 0.2);
    app.modelSettings[percent] = 80;
    app.onMoeExpertOffloadResidentPercent();
    await app.saveModelSettings();
    assert.equal(payload[fraction], 0.8);
    // The selector is gone: only the bounded percentage field remains.
    assert.ok(!/<select/.test(offload), 'the preset selector must be gone');
    assert.ok(
        /<input type="number" min="20" max="80" step="1"/.test(offload),
        'the percentage field must span two digits',
    );
    assert.ok(offload.includes('>%</span>'), 'the field must be marked with %');
    console.log('PASS: offload save/reopen, spec toggle exclusion and bounded resident fraction');
})().catch(error => {console.error(error); process.exitCode = 1});
