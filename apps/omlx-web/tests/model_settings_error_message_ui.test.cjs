// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(
    path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8'
);
const context = {
    localStorage: {getItem: () => null},
    window: {t: key => key},
    navigator: {language: 'en'},
    document: {},
    console,
};
const create = vm.runInNewContext(source + '\n dashboard;', context);
const app = create();

// A pydantic validation failure, as PUT /admin/api/models/{id}/settings
// answers it: alerting this array used to print "[object Object]".
const detail = [{
    type: 'value_error',
    loc: ['body', 'specprefill_draft_model'],
    msg: 'Value error, Draft model has no config.json: /gone/draft',
    input: '/gone/draft',
}];

const message = app.formatApiError(detail);
assert.ok(!message.includes('[object Object]'), message);
assert.equal(
    message,
    'specprefill_draft_model: Draft model has no config.json: /gone/draft'
);
assert.equal(app.formatApiError([{loc: [], msg: 'bare'}]), 'bare');
assert.equal(
    app.formatApiError([
        {loc: ['body', 'a'], msg: 'first'},
        {loc: ['body', 'b'], msg: 'second'},
    ]),
    'a: first; b: second'
);

// String, object and absent details still reach the i18n fallback intact.
assert.equal(app.formatApiError('Model not loaded'), 'Model not loaded');
assert.equal(app.formatApiError({message: 'nope'}), 'nope');
assert.equal(app.formatApiError(undefined), '');
assert.equal(app.formatApiError(null), '');

(async () => {
    app.loadModels = async () => {};
    app.selectedModel = {id: 'model-a', config_model_type: 'qwen3_5'};
    app.modelSettings = app.buildModelSettingsState(app.selectedModel, {});
    let alerted = null;
    context.alert = value => { alerted = value; };
    context.fetch = async () => ({
        ok: false,
        status: 422,
        json: async () => ({detail}),
    });

    await app.saveModelSettings();

    assert.equal(
        alerted,
        'specprefill_draft_model: Draft model has no config.json: /gone/draft'
    );
    console.log('Model settings save names the refused field instead of [object Object]');
})();
