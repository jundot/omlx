// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const root = path.join(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'omlx_web/static/js/dashboard.js'), 'utf8');
const context = {
    localStorage: {getItem: () => null}, window: {t: key => key}, console, URL,
    alert: message => {throw Error(message)},
};
const app = vm.runInNewContext(source + '\n dashboard;', context)();

// A link copied from a model page reduces to the owner/repo id the downloader needs.
[
    ['https://huggingface.co/mlx-community/Llama-3-8B-4bit', 'mlx-community/Llama-3-8B-4bit'],
    ['https://huggingface.co/mlx-community/Llama-3-8B-4bit/tree/main?tab=files', 'mlx-community/Llama-3-8B-4bit'],
    ['http://www.huggingface.co/mlx-community/Llama-3-8B-4bit', 'mlx-community/Llama-3-8B-4bit'],
    ['//huggingface.co/mlx-community/Llama-3-8B-4bit', 'mlx-community/Llama-3-8B-4bit'],
    ['hf.co/mlx-community/Llama-3-8B-4bit', 'mlx-community/Llama-3-8B-4bit'],
    // Anything else is passed through as typed, including a plain id.
    ['  mlx-community/Llama-3-8B-4bit  ', 'mlx-community/Llama-3-8B-4bit'],
    ['john.smith/my-model', 'john.smith/my-model'],
    ['https://github.com/mlx-project/mlx-models', 'https://github.com/mlx-project/mlx-models'],
    ['', ''],
].forEach(([input, expected]) => assert.equal(
    app.hfRepoIdFromInput(input), expected, `hfRepoIdFromInput(${JSON.stringify(input)})`));

// Enter and Download use the normalized id for the model lookup and field value.
let opened;
app.openModelDetail = (repoId, tab) => { opened = [repoId, tab]; };
app.hfRepoId = 'https://huggingface.co/mlx-community/Llama-3-8B-4bit/tree/main';
app.startHFDownload();
assert.deepEqual(opened, ['mlx-community/Llama-3-8B-4bit', 'files']);
assert.equal(app.hfRepoId, 'mlx-community/Llama-3-8B-4bit');

// A configured mirror URL, including an endpoint path prefix, is reduced the same way.
app.globalSettings.huggingface.endpoint = 'https://mirror.example:8443/hf';
[
    ['https://mirror.example:8443/hf/owner/repo/tree/main', 'owner/repo'],
    ['mirror.example:8443/hf/owner/repo', 'owner/repo'],
    ['https://mirror.example:8443/owner/repo', 'owner/repo'],
].forEach(([input, expected]) => assert.equal(
    app.hfRepoIdFromInput(input), expected, `hfRepoIdFromInput(${JSON.stringify(input)})`));
