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
vm.runInContext(fs.readFileSync(path.join(root, 'omlx_web/static/js/dashboard.js'), 'utf8'), context);
(async () => {
    const app = context.dashboard();
    app.loadModels = async () => {};
    let payload;
    context.fetch = async (url, init) => {
        payload = JSON.parse(init.body);
        assert.equal(init.method, 'PUT');
        return {ok: true, json: async () => ({})};
    };
    // Qwen4 PLE and DeepSeek V4.1 Engram share the SSD N-gram row and its save rule.
    for (const [key, modelType] of [['deepseek_v41_engram_ssd_offload', 'deepseek_v41'],
                                    ['qwen4_ple_ssd_offload', 'qwen4_exp']]) {
        for (const [forced, saved] of [[false,false], [false,true], [true,false], [true,true]]) {
            const model = {id:modelType, config_model_type:modelType, [key+'_supported']:true, [key+'_forced']:forced};
            app.selectedModel = model;
            app.modelSettings = app.buildModelSettingsState(model, {[key]:saved});
            assert.equal(app.modelSettings[key], forced || saved);
            assert.equal(app.modelSettings[key+'_forced'], forced);
            const html = fs.readFileSync(path.join(root,
                'omlx_web/templates/dashboard/_modal_model_settings.html'), 'utf8');
            const section = html.split('<!-- SSD N-gram Offload (Qwen4 PLE or DeepSeek V4.1 Engram) -->')[1]
                .split('<!-- MoE Expert Offload -->')[0];
            const click = section.match(/@click="([^"]+)"/)[1];
            const disabled = section.match(/:disabled="([^"]+)"/)[1];
            const scope = {modelSettings: app.modelSettings, ssdNgramOffloadKey: () => app.ssdNgramOffloadKey()};
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
    }
    console.log('PASS: forced/saved/toggled/other-model modal state and actual PUT payload');
})().catch(error => {console.error(error); process.exitCode = 1});

{
    const app = context.dashboard();
    app.oqModels = [
        {path: 'v41', model_type: 'deepseek_v41'},
        {path: 'llama', model_type: 'llama'},
    ];
    const html = fs.readFileSync(path.join(root,
        'omlx_web/templates/dashboard/_models.html'), 'utf8');
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
    assert.deepEqual(Array.from(app.oqAvailableLevels()), [2, 2.5, 2.7, 3, 4]);
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

(async () => {
    // A separate context keeps this fetch stub away from the modal test above.
    const oqContext = {localStorage: {getItem: () => null}, window: {t: k => k}, console};
    vm.createContext(oqContext);
    vm.runInContext(fs.readFileSync(path.join(root, 'omlx_web/static/js/dashboard.js'), 'utf8'), oqContext);
    const app = oqContext.dashboard();
    app.oqModels = [
        {path: 'v41', model_type: 'deepseek_v41'},
        {path: 'llama', model_type: 'llama'},
    ];
    const payloads = [];
    oqContext.fetch = async (_url, init) => {
        payloads.push(JSON.parse(init.body));
        return {ok: false, json: async () => ({detail: 'stop'})};
    };
    app.oqPreserveNgram = true;
    for (const model of ['v41', 'llama']) {
        app.oqSelectedModelPath = model;
        await app.startOQQuantization();
    }
    assert.deepEqual(payloads.map(p => p.preserve_ngram_table), [true, false]);
    const html = fs.readFileSync(path.join(root,
        'omlx_web/templates/dashboard/_models.html'), 'utf8');
    assert.match(html.match(/x-effect="([^"\n]*oqApplyModelPolicy[^"\n]*)"/)[1], /oqPreserveNgram/);
    console.log('PASS: Engram toggle reaches the oQ start payload for DeepSeek V4.1 only');
})().catch(error => {console.error(error); process.exitCode = 1});

(async () => {
    let pending;
    const queries = [];
    const alpineContext = vm.createContext({
        window: {t: k => k, Element: class {}},
        document: {createElement: () => ({})},
        MutationObserver: class {},
        queueMicrotask: () => {},
        localStorage: {getItem: () => null},
        URLSearchParams,
        console,
        setTimeout: callback => { pending = callback; return 1; },
        clearTimeout: () => { pending = null; },
        fetch: async url => {
            const query = Object.fromEntries(new URLSearchParams(url.split('?')[1]));
            queries.push(query);
            if (query.mtp_assistant_model_path === 'incompatible') {
                return {ok: false, json: async () => ({detail: 'Donor mismatch'})};
            }
            return {ok: true, json: async () => ({output_size_bytes: queries.length})};
        },
    });
    const js = path.join(root, 'omlx_web/static/js');
    vm.runInContext(fs.readFileSync(path.join(js, 'alpine.min.js'), 'utf8'), alpineContext);
    alpineContext.queueMicrotask = queueMicrotask;
    vm.runInContext(fs.readFileSync(path.join(js, 'dashboard.js'), 'utf8'), alpineContext);
    const app = alpineContext.window.Alpine.reactive(alpineContext.dashboard());
    app.oqModels = [{path: 'source', model_type: 'qwen3_5', has_mtp_heads: true}];
    app.oqAllModels = ['donor', 'incompatible'].map(path => ({
        path, model_type: 'qwen3_5', has_mtp_heads: true,
    }));
    app.oqSelectedModelPath = 'source';
    const html = fs.readFileSync(path.join(root,
        'omlx_web/templates/dashboard/_models.html'), 'utf8');
    const effect = html.match(/x-effect="([^"\n]*oqRefreshEstimate[^"\n]*)"/)[1];
    const runEffect = vm.runInNewContext(`(app) => { with (app) { ${effect} } }`);
    alpineContext.window.Alpine.effect(() => runEffect(app));
    async function refreshed() {
        await new Promise(setImmediate);
        assert.equal(typeof pending, 'function', 'Option change must refresh the estimate');
        const run = pending;
        pending = null;
        await run();
        return queries.at(-1);
    }

    await refreshed();
    const mtpClick = html.match(/@click="([^"\n]*oqPreserveMtp = !oqPreserveMtp)"/)[1];
    vm.runInNewContext(`with (app) { ${mtpClick} }`, {app});
    assert.equal((await refreshed()).preserve_mtp, 'true');
    app.oqPreserveMtp = false;
    app.oqDtype = 'float16';
    assert.equal((await refreshed()).dtype, 'float16');
    app.oqTextOnly = true;
    assert.equal((await refreshed()).text_only, 'true');
    app.oqMtpAssistantPath = 'incompatible';
    await refreshed();
    assert.equal(app.oqEstimate, null);
    assert.equal(app.oqEstimateError, 'Donor mismatch');
    app.oqMtpAssistantPath = 'donor';
    assert.equal((await refreshed()).mtp_assistant_model_path, 'donor');
    assert.equal(app.oqEstimateError, '');
    console.log('PASS: every oQ output option refreshes the size estimate');
})().catch(error => {console.error(error); process.exitCode = 1});
