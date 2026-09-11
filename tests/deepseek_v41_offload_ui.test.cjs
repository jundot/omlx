// SPDX-License-Identifier: Apache-2.0
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
(async () => {
    const app = context.dashboard();
    app.loadModels = async () => {};
    let payload;
    context.fetch = async (url, init) => {
        payload = JSON.parse(init.body);
        assert.equal(init.method, 'PUT');
        return {ok: true, json: async () => ({})};
    };
    const key = 'deepseek_v41_engram_ssd_offload';
    for (const [forced, saved] of [[false,false], [false,true], [true,false], [true,true]]) {
        const model = {id:'v41', config_model_type:'deepseek_v41', [key+'_supported']:true, [key+'_forced']:forced};
        app.selectedModel = model;
        app.modelSettings = app.buildModelSettingsState(model, {[key]:saved});
        assert.equal(app.modelSettings[key], forced || saved);
        assert.equal(app.modelSettings[key+'_forced'], forced);
        const html = fs.readFileSync(path.join(root,
            'omlx/admin/templates/dashboard/_modal_model_settings.html'), 'utf8');
        const section = html.split('<!-- DeepSeek V4.1 Engram SSD Offload -->')[1]
            .split('<!-- Thinking Budget -->')[0];
        const click = section.match(/@click="([^"]+)"/)[1];
        const disabled = section.match(/:disabled="([^"]+)"/)[1];
        const scope = {modelSettings: app.modelSettings};
        assert.equal(vm.runInNewContext(disabled, scope), forced);
        vm.runInNewContext(click, scope);
        assert.equal(app.modelSettings[key], forced || !saved);
        if (!forced) vm.runInNewContext(click, scope);
        await app.saveModelSettings();
        assert.equal(payload[key], saved, 'Saving unrelated settings must preserve requested storage mode');
    }
    app.selectedModel[key+'_forced'] = false;
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {[key]:false});
    app.modelSettings[key] = true;
    await app.saveModelSettings();
    assert.equal(payload[key], true);
    const other = app.buildModelSettingsState({config_model_type:'llama'}, {});
    assert.equal(other[key+'_supported'], false);
    console.log('PASS: forced/saved/toggled/other-model modal state and actual PUT payload');
})().catch(error => {console.error(error); process.exitCode = 1});
