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
    console.log('PASS: offload save/reopen and bidirectional speculative toggle exclusion');
})().catch(error => {console.error(error); process.exitCode = 1});
