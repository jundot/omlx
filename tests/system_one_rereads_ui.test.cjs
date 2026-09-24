// SPDX-License-Identifier: Apache-2.0
// The System One re-read toggle: shown and saved for diffusion models only.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const root = path.resolve(__dirname, '..');
const context = {
    localStorage: {getItem: () => null},
    window: {t: k => k},
    console,
    alert: message => {throw Error(message)},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'omlx/admin/static/js/dashboard.js'), 'utf8'), context);

const key = 'system_one_rereads_enabled';
const diffusion = {id: 'dgemma', config_model_type: 'diffusion_gemma'};
const llama = {id: 'llama', config_model_type: 'llama'};
const html = fs.readFileSync(path.join(root,
    'omlx/admin/templates/dashboard/_modal_model_settings.html'), 'utf8');
const section = html.split('<!-- System One re-reads (diffusion models only) -->')[1]
    .split('<!-- Trust Remote Code')[0];
const show = section.match(/x-show="([^"]+)"/)[1];
const click = section.match(/@click="([^"]+)"/)[1];

(async () => {
    const app = context.dashboard();
    app.loadModels = async () => {};
    let payload;
    context.fetch = async (url, init) => {
        payload = JSON.parse(init.body);
        return {ok: true, json: async () => ({})};
    };

    // Diffusion model: on by default, loads a saved "off", toggles, saves.
    app.selectedModel = diffusion;
    app.modelSettings = app.buildModelSettingsState(diffusion, {});
    assert.equal(app.modelSettings[key], true);
    app.modelSettings = app.buildModelSettingsState(diffusion, {[key]: false});
    assert.equal(app.modelSettings[key], false);
    const scope = {modelSettings: app.modelSettings};
    assert.equal(vm.runInNewContext(show, scope), true);
    vm.runInNewContext(click, scope);
    assert.equal(app.modelSettings[key], true);
    vm.runInNewContext(click, scope);
    await app.saveModelSettings();
    assert.equal(payload[key], false);

    // Profiles carry it for diffusion models; a preset resets it to on.
    app.profileFields = {universal: [key], model_specific: []};
    assert.equal(app.formValuesForProfile()[key], false);
    app._resetPresetApplicableFields();
    assert.equal(app.modelSettings[key], true);

    // Any other model neither shows, saves nor profiles it.
    app.selectedModel = llama;
    app.modelSettings = app.buildModelSettingsState(llama, {[key]: false});
    assert.equal(app.modelSettings[key], null);
    assert.equal(vm.runInNewContext(show, {modelSettings: app.modelSettings}), false);
    await app.saveModelSettings();
    assert.equal(key in payload, false);
    assert.equal(key in app.formValuesForProfile(), false);
    console.log('PASS: diffusion-only toggle state, visibility, PUT payload and profile values');
})().catch(error => {console.error(error); process.exitCode = 1});
