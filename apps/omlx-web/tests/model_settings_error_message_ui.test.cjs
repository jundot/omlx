// SPDX-License-Identifier: Apache-2.0
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const context = {localStorage: {getItem: () => null}, window: {t: k => k}, console};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8'), context);
(async () => {
    // A pydantic 422 is an array of {loc, msg}; the profile error used to show "[object Object]".
    const app = context.dashboard();
    app.selectedModel = {id: 'model-a'};
    app.newProfile = {display_name: 'Draft profile', api_name: 'draft_profile'};
    app.formValuesForProfile = () => ({});
    context.fetch = async () => ({ok: false, status: 422, json: async () => ({detail: [
        {loc: ['body', 'settings', 'specprefill_draft_model'], msg: 'Value error, Draft model has no config.json: /gone/draft'},
        {loc: ['body', 'api_name'], msg: 'too long'},
    ]})});
    await app.createProfile();
    assert.equal(app.profileError,
        'settings.specprefill_draft_model: Draft model has no config.json: /gone/draft; api_name: too long');
})().catch(error => { console.error(error); process.exitCode = 1; });
