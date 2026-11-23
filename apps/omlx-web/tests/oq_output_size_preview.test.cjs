// SPDX-License-Identifier: Apache-2.0
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const root = path.resolve(__dirname, '..');
(async () => {
    // Use the bundled Alpine reactivity engine. Calling the effect explicitly
    // after a click would miss the regression in its dependency tracking.
    let pendingEstimate;
    const estimates = [];
    let started;
    const ctx = vm.createContext({
        window: {t: key => key, Element: class {}},
        document: {createElement: () => ({})},
        MutationObserver: class {},
        queueMicrotask: () => {}, // Skip Alpine's DOM startup.
        localStorage: {getItem: () => null},
        URLSearchParams,
        console,
        setTimeout: callback => { pendingEstimate = callback; return 1; },
        clearTimeout: () => { pendingEstimate = null; },
        fetch: async (url, init) => {
            if (init) {
                started = JSON.parse(init.body);
                return {ok: false, json: async () => ({detail: 'stop'})};
            }
            const query = Object.fromEntries(new URLSearchParams(url.split('?')[1]));
            estimates.push(query);
            if (query.mtp_assistant_model_path === 'incompatible') {
                return {ok: false, json: async () => ({detail: 'Donor mismatch'})};
            }
            const bytes = 100 + (query.preserve_mtp === 'true' ? 8 : 0)
                + (query.text_only === 'true' ? 0 : 16)
                + (query.mtp_assistant_model_path ? 4 : 0)
                + (query.dtype === 'float16' && query.text_only !== 'true' ? 16 : 0);
            return {ok: true, json: async () => ({output_size_bytes: bytes})};
        },
    });
    const assets = path.join(root, 'omlx_web/static/js');
    vm.runInContext(fs.readFileSync(path.join(assets, 'alpine.min.js'), 'utf8'), ctx);
    ctx.queueMicrotask = queueMicrotask;
    vm.runInContext(fs.readFileSync(path.join(assets, 'dashboard.js'), 'utf8'), ctx);
    const app = ctx.window.Alpine.reactive(ctx.dashboard());
    app.oqModels = [{path: 'source', model_type: 'qwen3_5', has_mtp_heads: true}];
    app.oqAllModels = ['donor', 'incompatible'].map(path => ({
        path, model_type: 'qwen3_5', has_mtp_heads: true,
    }));
    app.oqSelectedModelPath = 'source';
    const html = fs.readFileSync(path.join(root,
        'omlx_web/templates/dashboard/_models.html'), 'utf8');
    const expr = html.match(/x-effect="([^"\n]*oqRefreshEstimate[^"\n]*)"/)[1];
    const runEffect = vm.runInNewContext('(app) => { with (app) { ' + expr + ' } }');
    ctx.window.Alpine.effect(() => runEffect(app));

    async function finishEstimate(size) {
        await new Promise(setImmediate);
        assert.equal(typeof pendingEstimate, 'function', 'Option change must refresh the estimate');
        const run = pendingEstimate;
        pendingEstimate = null;
        await run();
        assert.equal(app.oqEstimate?.output_size_bytes, size);
    }

    await finishEstimate(116);
    const mtpClick = html.match(/@click="([^"\n]*oqPreserveMtp = !oqPreserveMtp)"/)[1];
    vm.runInNewContext('with (app) { ' + mtpClick + ' }', {app});
    await finishEstimate(124);
    app.oqPreserveMtp = false;
    await finishEstimate(116);
    app.oqDtype = 'float16';
    await finishEstimate(132);
    app.oqTextOnly = true;
    await finishEstimate(100);
    app.oqMtpAssistantPath = 'donor';
    await finishEstimate(104);

    await app.startOQQuantization();
    const fromStart = Object.fromEntries(Object.entries(started)
        .filter(([key]) => key in estimates.at(-1))
        .map(([key, value]) => [key, String(value)]));
    assert.deepEqual(fromStart, estimates.at(-1));
    assert.ok(estimates.every(query => query.oq_level === '4'));

    app.oqMtpAssistantPath = 'incompatible';
    await finishEstimate(undefined);
    assert.equal(app.oqEstimateError, 'Donor mismatch');
    app.oqMtpAssistantPath = '';
    await finishEstimate(100);
    assert.equal(app.oqEstimateError, '');
})().catch(error => {console.error(error.message); process.exitCode = 1});
