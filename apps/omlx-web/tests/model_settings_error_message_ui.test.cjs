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

(async () => {
    // Profile create answers a pydantic validation failure with an array of
    // {loc, msg} objects; the profile error used to show "[object Object]".
    app.selectedModel = {id: 'model-a'};
    app.newProfile = {
        display_name: 'Draft profile',
        api_name: 'draft_profile',
        description: '',
        expose_as_model: false,
    };
    app.formValuesForProfile = () => ({});
    context.fetch = async () => ({
        ok: false,
        status: 422,
        json: async () => ({
            detail: [
                {
                    type: 'value_error',
                    loc: ['body', 'settings', 'specprefill_draft_model'],
                    msg: 'Value error, Draft model has no config.json: /gone/draft',
                },
                {loc: ['body', 'api_name'], msg: 'too long'},
            ],
        }),
    });

    await app.createProfile();

    assert.equal(
        app.profileError,
        'settings.specprefill_draft_model: Draft model has no config.json: /gone/draft; '
        + 'api_name: too long'
    );
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
