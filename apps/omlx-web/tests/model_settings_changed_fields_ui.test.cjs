// SPDX-License-Identifier: Apache-2.0
// #3089: the model-settings modal must submit only the fields the user changed.
// The backend applies each key present in the request body
// (omlx/admin/routes.py uses request.model_fields_set), so a field the user never
// touched gets written back with whatever the form happened to hold.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const root = path.resolve(__dirname, '..');

function loadDashboard() {
    const context = {
        localStorage: {getItem: () => null},
        window: {t: k => k},
        console,
        alert: message => {throw Error(message)},
    };
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(
        path.join(root, 'omlx_web/static/js/dashboard.js'), 'utf8'), context);
    const app = context.dashboard();
    app.loadModels = async () => {};
    let body = null;
    context.fetch = async (url, init) => {
        body = JSON.parse(init.body);
        assert.equal(init.method, 'PUT');
        return {ok: true, json: async () => ({})};
    };
    return {app, submitted: () => body};
}

// A saved state plus the form state it produces, as openModelSettings would.
function openModal(saved) {
    const {app, submitted} = loadDashboard();
    const model = {id: 'model-a', config_model_type: 'qwen3'};
    app.selectedModel = model;
    app.modelSettings = app.buildModelSettingsState(model, saved);
    app._modelSettingsBaseline = JSON.stringify(app.modelSettings);
    return {app, submitted};
}

(async () => {
    // A feature switched on outside the modal (hand-edited model_settings.json,
    // a preset, or a CLI flag) must survive an unrelated save.
    const untouched = ['mtp_enabled', 'vlm_mtp_enabled', 'turboquant_kv_enabled',
                       'dflash_enabled', 'specprefill_enabled', 'qwen35_oq_a8_enabled',
                       'moe_expert_offload_enabled', 'guided_grammar_enabled',
                       'force_sampling', 'trust_remote_code'];
    {
        const {app, submitted} = openModal({
            mtp_enabled: true, vlm_mtp_enabled: true, turboquant_kv_enabled: true,
            dflash_enabled: true, specprefill_enabled: true, qwen35_oq_a8_enabled: true,
            moe_expert_offload_enabled: true, guided_grammar_enabled: true,
            force_sampling: true, trust_remote_code: true, temperature: 0.5,
        });
        app.modelSettings.temperature = 0.9;
        await app.saveModelSettings();
        const body = submitted();
        assert.equal(body.temperature, 0.9);
        for (const key of untouched) {
            assert.equal(key in body, false,
                `${key} was not changed, so saving temperature must not resubmit it`);
        }
    }

    // A field the user did change must still reach the server, in both directions.
    for (const [saved, turnedOn] of [[true, false], [false, true]]) {
        const {app, submitted} = openModal({mtp_enabled: saved});
        app.modelSettings.mtp_enabled = turnedOn;
        await app.saveModelSettings();
        assert.equal(submitted().mtp_enabled, turnedOn);
    }

    // An untouched derived toggle must not turn a feature on either: the budget
    // form value lives in enableThinkingBudget, not in thinking_budget_enabled.
    {
        const {app, submitted} = openModal({});
        app.modelSettings.temperature = 0.9;
        await app.saveModelSettings();
        assert.equal('thinking_budget_enabled' in submitted(), false);
    }
    {
        const {app, submitted} = openModal({});
        app.modelSettings.enableThinkingBudget = true;
        app.modelSettings.thinking_budget_tokens = 512;
        await app.saveModelSettings();
        assert.equal(submitted().thinking_budget_enabled, true);
        assert.equal(submitted().thinking_budget_tokens, 512);
    }

    // Numeric fields are held as strings in the form but submitted as numbers,
    // so an untouched depth must not be resubmitted as a changed value.
    {
        const {app, submitted} = openModal({mtp_enabled: true, mtp_adaptive_max_depth: 5});
        app.modelSettings.temperature = 0.9;
        await app.saveModelSettings();
        assert.equal('mtp_adaptive_max_depth' in submitted(), false);
    }
    {
        const {app, submitted} = openModal({mtp_enabled: true, mtp_adaptive_max_depth: 5});
        app.modelSettings.mtp_adaptive_max_depth = '6';
        await app.saveModelSettings();
        assert.equal(submitted().mtp_adaptive_max_depth, 6);
    }

    // Without a baseline snapshot (a modal opened before this behaviour existed)
    // the save must still work rather than throw.
    {
        const {app, submitted} = loadDashboard();
        const model = {id: 'model-a', config_model_type: 'qwen3'};
        app.selectedModel = model;
        app.modelSettings = app.buildModelSettingsState(model, {mtp_enabled: true});
        await app.saveModelSettings();
        assert.equal(submitted().mtp_enabled, true);
    }
})().catch(e => {
    console.error(e);
    process.exit(1);
});
