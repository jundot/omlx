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
    // Unified contract: saving migrates legacy keys to canonical ones.
    const fraction = 'moe_expert_offload_resident_fraction';
    for (const value of [.125, .25, .5, .75]) {
        app.modelSettings = app.buildModelSettingsState(app.selectedModel, {expert_streaming_enabled: true, [fraction]: value});
        await app.saveModelSettings();
        assert.equal(payload.expert_streaming_enabled, true);
        assert.equal(payload.moe_expert_offload_enabled, false);
        assert.equal(payload[fraction], value);
        assert.equal(payload.expert_streaming_budget_auto, true);
        assert.equal(payload.expert_streaming_dynamic, null);
    }
    // Reopen with legacy keys: effective enable ORs the alias.
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {moe_expert_offload_enabled: true});
    assert.equal(app.modelSettings.expert_streaming_enabled, true);
    assert.equal(app.modelSettings.moe_expert_offload_enabled, true);
    const html = fs.readFileSync(path.join(root, 'omlx/admin/templates/dashboard/_modal_model_settings.html'), 'utf8');
    const offload = html.split('<!-- Expert streaming (unified backend')[1].split('<!-- IndexCache')[0];
    const condition = offload.match(/x-if="([^"]+)"/)[1];
    const scope = (supported, streaming) => ({selectedModel: {moe_expert_offload_supported: supported}, modelSettings: {expert_streaming_supported: streaming}});
    assert.equal(vm.runInNewContext(condition, scope(true, false)), true);
    assert.equal(vm.runInNewContext(condition, scope(false, true)), true);
    assert.equal(vm.runInNewContext(condition, scope(false, false)), false);
    assert.ok(!vm.runInNewContext(condition, scope(undefined, undefined)));
    // The modal's :disabled expressions call helpers on the app instance —
    // bind the REAL ones (expertOffloadEffective, isDeepseekV41Model,
    // vlmMtpProcessorConflict) so the expressions evaluate exactly as in
    // the live component rather than against hand-rolled stubs.
    const baseScope = (modelSettings, selectedModel) => ({
        modelSettings,
        selectedModel: selectedModel || {},
        expertOffloadEffective: () => app.expertOffloadEffective.call({modelSettings}),
        isDeepseekV41Model: m => app.isDeepseekV41Model(m),
        vlmMtpProcessorConflict: () => app.vlmMtpProcessorConflict.call({modelSettings}),
    });
    const enabled = 'expert_streaming_enabled';
    for (const key of ['mtp_enabled', 'vlm_mtp_enabled', 'dflash_enabled']) {
        // The expert-streaming toggle disables while another speculative
        // path is on — evaluated against the REAL helper now.
        const disScope = baseScope({[enabled]: false, [key]: true});
        assert.equal(vm.runInNewContext(offload.match(/:disabled="([^"]+)"/)[1], disScope), true);
        const lines = html.split('\n');
        const needle = 'modelSettings.' + key + ' = !modelSettings.' + key;
        const i = lines.findIndex(line => line.includes('@click=') && line.includes(needle));
        const disabled = lines[i + 1].match(/:disabled="([^"]+)"/)[1];
        // Reciprocal direction: enabling expert streaming disables the
        // other speculative toggles. mtp carries the deepseek_v41
        // exemption, so it needs a non-v41 selectedModel to disable.
        const scope = baseScope(
            {moe_expert_offload_enabled: true},
            {id: 'qwen', config_model_type: 'qwen3_moe'}
        );
        assert.equal(vm.runInNewContext(disabled, scope), true, `${key} disabled expr`);
    }
    // deepseek_v41 Lightning-MTP exemption: on a v41 model the mtp toggle
    // is NOT disabled by expert offload (the backend permits the pair —
    // DSpark verify runs under frozen residency).
    {
        const lines = html.split('\n');
        const needle = 'modelSettings.mtp_enabled = !modelSettings.mtp_enabled';
        const i = lines.findIndex(line => line.includes('@click=') && line.includes(needle));
        const disabled = lines[i + 1].match(/:disabled="([^"]+)"/)[1];
        const v41Scope = baseScope(
            {moe_expert_offload_enabled: true, mtp_compatible: true},
            {id: 'v41', config_model_type: 'deepseek_v41'}
        );
        assert.ok(!vm.runInNewContext(disabled, v41Scope),
            'deepseek_v41 exempts mtp from the offload gate');
        // But DFlash and VLM-MTP still lock on v41 (no exemption there).
        for (const key of ['vlm_mtp_enabled', 'dflash_enabled']) {
            const j = lines.findIndex(line =>
                line.includes('@click=') && line.includes('modelSettings.' + key + ' = !modelSettings.' + key));
            const dis = lines[j + 1].match(/:disabled="([^"]+)"/)[1];
            assert.equal(vm.runInNewContext(dis, v41Scope), true,
                `${key} stays disabled under offload on v41`);
        }
    }
    console.log('PASS: unified streaming save/migrate/reopen, bidirectional speculative exclusion, and the v41 mtp exemption');
})();
