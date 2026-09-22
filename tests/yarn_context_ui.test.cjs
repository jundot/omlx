// SPDX-License-Identifier: Apache-2.0
// Dashboard contract for single-knob YaRN: Max Context Window doubles as the
// rope-scaling target on qwen4_exp models. Covers the terse read-only factor
// readout, the 4x client validator, payload hygiene (no separate field), and
// the gated hint under the Ctx Window input.
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
    app.selectedModel = {
        id: 'q4',
        config_model_type: 'qwen4_exp',
        yarn_rope_supported: true,
        model_context_length: 262144,
    };
    let payload;
    context.fetch = async (_, init) => {
        payload = JSON.parse(init.body);
        return {ok: true, json: async () => ({})};
    };

    // Terse factor readout mirrors the loader derivation.
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {});
    assert.equal(app.yarnFactorDisplay(), 'off');
    app.modelSettings.max_context_window = 131072;
    assert.equal(app.yarnFactorDisplay(), 'off');
    app.modelSettings.max_context_window = 524288;
    assert.equal(app.yarnFactorDisplay(), '2 (524,288 / 262,144)');
    app.modelSettings.max_context_window = 393216;
    assert.equal(app.yarnFactorDisplay(), '1.5 (393,216 / 262,144)');
    app.modelSettings.max_context_window = 262144 * 4 + 1;
    assert.match(app.yarnFactorDisplay(), /\d \(1,048,577 \/ 262,144\) — over 4× cap/);

    // Validator: 4x cap only; above-native values are the feature, not an error.
    app.modelSettings.max_context_window = 524288;
    assert.equal(app.validateYarnSettings(), null);
    app.modelSettings.max_context_window = 262144 * 4 + 1;
    assert.match(app.validateYarnSettings(), /4x the native context/);
    // Save aborts on the over-cap value (alert throws in this context).
    let alerted = null;
    try {
        await app.saveModelSettings();
    } catch (error) {
        alerted = error.message;
    }
    assert.match(String(alerted), /4x the native context/);

    // Valid value saves through the plain max_context_window field — and the
    // retired separate setting must not reappear anywhere.
    app.modelSettings.max_context_window = 524288;
    await app.saveModelSettings();
    assert.equal(payload.max_context_window, 524288);
    assert.equal('yarn_context_length' in payload, false);
    assert.equal('yarn_context_length' in app.buildModelSettingsState(app.selectedModel, {yarn_context_length: 524288}), false);

    // Non-qwen4 models: no display, no cap.
    const other = {id: 'x', config_model_type: 'llama', yarn_rope_supported: false, model_context_length: 131072};
    app.selectedModel = other;
    app.modelSettings = app.buildModelSettingsState(other, {max_context_window: 10000000});
    assert.equal(app.yarnFactorDisplay(), '');
    assert.equal(app.validateYarnSettings(), null);

    // The readout lives under the Ctx Window input, is flag-gated, and the
    // formula disclosure is gone (terse-by-design).
    const html = fs.readFileSync(
        path.join(root, 'omlx/admin/templates/dashboard/_modal_model_settings.html'), 'utf8');
    assert.match(html, /x-show="modelSettings\.yarn_rope_supported"/);
    assert.match(html, /x-text="yarnFactorDisplay\(\)"/);
    assert.match(html, /modal\.model_settings\.yarn_factor_label/);
    assert.equal(html.includes('yarnInfoOpen'), false);
    assert.equal(html.includes('yarn_factor_formula'), false);
    assert.equal(html.includes('yarn_context_length'), false);

    console.log('PASS: single-knob YaRN terse readout, validator, payload, and gated hint');
})().catch(error => {console.error(error); process.exitCode = 1});
