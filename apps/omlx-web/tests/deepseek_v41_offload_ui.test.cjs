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
    let estimateCallback;
    let firstEstimate;
    const requests = [];
    const locales = path.join(root, 'omlx_web/i18n');
    const loadCatalog = file => JSON.parse(fs.readFileSync(path.join(locales, file), 'utf8'));
    for (const file of fs.readdirSync(locales).filter(file => file.endsWith('.json'))) {
        const catalog = loadCatalog(file);
        assert.match(catalog['models.oq.engramPrecisionAutoBits'], /\{bits\}/);
        assert.ok(catalog['models.oq.engramPrecisionHelp']);
    }
    let catalog = loadCatalog('en.json');
    let lastEstimateOptions, lastStartOptions;
    const oqContext = {
        localStorage: {getItem: () => null},
        window: {t: key => catalog[key] ?? key},
        console,
        URLSearchParams,
        setTimeout: fn => { estimateCallback = fn; return 1; },
        clearTimeout: () => {},
        fetch: async (url, init) => {
            if (init) {
                lastStartOptions = JSON.parse(init.body);
                requests.push(lastStartOptions.engram_precision);
                return {ok: false, json: async () => ({detail: 'stop'})};
            }
            lastEstimateOptions = Object.fromEntries(new URLSearchParams(url.split('?')[1]));
            const precision = lastEstimateOptions.engram_precision;
            if (precision === 'auto') {
                return new Promise(resolve => { firstEstimate = resolve; });
            }
            return {ok: true, json: async () => ({output_size_bytes: 16})};
        },
    };
    vm.createContext(oqContext);
    vm.runInContext(fs.readFileSync(path.join(root, 'omlx_web/static/js/dashboard.js'), 'utf8'), oqContext);
    const app = oqContext.dashboard();
    app.oqModels = [
        {path: 'v41', model_type: 'deepseek_v41', engram_precision: {
            source_precision: 'fp8', available: ['auto', '8'],
        }},
        {path: 'hy4', model_type: 'hy_v4_flash', engram_precision: {
            source_precision: 'bf16', available: ['auto', '8', '16'],
            auto_bits: {'3.5': 4, 6: 6},
        }},
        {path: 'hy4-vl', model_type: 'hy_v4_vl'},
        {path: 'nvfp4', model_type: 'deepseek_v41', engram_precision: {
            source_precision: 'mxfp4', available: ['auto'],
        }},
    ];
    const html = fs.readFileSync(path.join(root,
        'omlx_web/templates/dashboard/_models.html'), 'utf8');
    const effect = html.match(/x-effect="([^"\n]*oqApplyModelPolicy[^"\n]*)"/)[1];
    assert.match(effect, /oqEngramPrecision/);

    app.oqSelectedModelPath = 'hy4';
    app.oqApplyModelPolicy();
    assert.deepEqual(Array.from(app.oqEngramPrecisionChoices(), c => c.value), ['auto', '8', '16']);
    app.oqLevel = 3.5;
    assert.equal(app.oqEngramPrecisionChoices()[0].label, 'Auto (4-bit)');
    app.oqLevel = 6;
    assert.equal(app.oqEngramPrecisionChoices()[0].label, 'Auto (6-bit)');
    catalog = loadCatalog('ko.json');
    app.oqLevel = 3.5;
    assert.equal(app.oqEngramPrecisionChoices()[0].label, '자동 (4비트)');
    catalog = loadCatalog('en.json');
    assert.match(app.oqEngramPrecisionChoices()[1].label, /MLX affine/);
    assert.match(app.oqEngramPrecisionChoices()[2].label, /BF16/);

    app.oqRefreshEstimate();
    const stale = estimateCallback();
    app.oqEngramPrecision = '16';
    app.oqDtype = 'float16';
    vm.runInNewContext('with (app) { ' + effect + ' }', {app});
    await estimateCallback();
    firstEstimate({ok: true, json: async () => ({output_size_bytes: 4})});
    await stale;
    assert.equal(app.oqEstimate.output_size_bytes, 16);
    await app.startOQQuantization();
    for (const [key, value] of Object.entries(lastEstimateOptions)) {
        assert.equal(String(lastStartOptions[key]), value, key);
    }
    assert.equal(lastEstimateOptions.engram_precision, '16');
    assert.equal(lastEstimateOptions.dtype, 'float16');
    assert.ok(!('preserve_ngram_table' in lastEstimateOptions));

    app.oqSelectedModelPath = 'v41';
    app.oqApplyModelPolicy();
    assert.equal(app.oqEngramPrecision, 'auto');
    assert.deepEqual(Array.from(app.oqEngramPrecisionChoices(), c => c.value), ['auto', '8']);
    assert.match(app.oqEngramPrecisionChoices()[1].label, /FP8/);
    app.oqEngramPrecision = '8';
    await app.startOQQuantization();
    app.oqEngramPrecision = '16';
    await app.startOQQuantization();

    for (const model of ['hy4-vl', 'nvfp4']) {
        app.oqSelectedModelPath = model;
        assert.deepEqual(Array.from(app.oqEngramPrecisionChoices(), c => c.value), ['auto']);
        await app.startOQQuantization();
    }
    assert.deepEqual(requests, ['16', '8', 'auto', 'auto', 'auto']);
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
