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

{
    const app = context.dashboard();
    app.oqModels = [
        {path: 'v41', model_type: 'deepseek_v41'},
        {path: 'llama', model_type: 'llama'},
    ];
    const html = fs.readFileSync(path.join(root,
        'omlx/admin/templates/dashboard/_models.html'), 'utf8');
    const effect = html.match(/x-effect="([^"\n]*oqApplyModelPolicy[^"\n]*)"/)[1];
    let estimatedLevel;
    app.oqRefreshEstimate = () => { estimatedLevel = app.oqLevel; };
    app.oqSelectedModelPath = 'llama';
    app.oqLevel = 8;
    app.oqDtype = 'float16';
    app.oqTextOnly = true;
    app.oqSensitivityModelPath = 'proxy';
    app.oqApplyModelPolicy();
    assert.equal(app.oqLevel, 8);
    assert.equal(app.oqDtype, 'float16');
    app.oqSelectedModelPath = 'v41';
    vm.runInNewContext(`with (app) { ${effect} }`, {app});
    assert.equal(estimatedLevel, 4);
    assert.deepEqual(Array.from(app.oqAvailableLevels()), [3, 4]);
    assert.equal(app.oqLevel, 4);
    assert.equal(app.oqDtype, 'bfloat16');
    assert.equal(app.oqTextOnly, false);
    assert.equal(app.oqSensitivityModelPath, '');
    app.oqLevel = 3;
    app.oqSensitivityModelPath = 'proxy';
    app.oqApplyModelPolicy();
    assert.equal(app.oqLevel, 3);
    assert.equal(app.oqSensitivityModelPath, 'proxy');
    app.oqSelectedModelPath = 'llama';
    assert.ok(app.oqAvailableLevels().includes(8));
}
