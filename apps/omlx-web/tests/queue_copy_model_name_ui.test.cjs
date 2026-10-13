const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');

const source = fs.readFileSync(
    path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8'
);

function stateWithModels(models) {
    const create = vm.runInNewContext(source + '\n dashboard;', {
        URL, console,
        localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => key, location: {reload() {}}},
        navigator: {language: 'en'}, document: {},
        setTimeout: () => {},
        fetch: async () => ({ok: true, json: async () => ({})}),
    });
    const state = create();
    state.loadStats = async () => {};
    state.loadModels = async () => {};
    state.models = models;
    return state;
}

test('queue copy matches the manager: loaded model copies its canonical id', () => {
    // On a real machine the API keys on the short folder id while the
    // download queue holds the full HuggingFace repo id.
    const state = stateWithModels([
        {id: 'Qwen3.5-9B-OptiQ-4bit', display_name: 'mlx-community/Qwen3.5-9B-OptiQ-4bit', settings: {}},
    ]);
    assert.equal(
        state.queueModelName('mlx-community/Qwen3.5-9B-OptiQ-4bit'),
        'Qwen3.5-9B-OptiQ-4bit'
    );
});

test('queue copy prefers a model alias like the manager does', () => {
    const state = stateWithModels([
        {id: 'Qwen3.5-9B-OptiQ-4bit', display_name: 'mlx-community/Qwen3.5-9B-OptiQ-4bit', settings: {model_alias: 'qwen-fast'}},
    ]);
    assert.equal(
        state.queueModelName('mlx-community/Qwen3.5-9B-OptiQ-4bit'),
        'qwen-fast'
    );
});

test('queue copy falls back to the folder name before the model loads', () => {
    const state = stateWithModels([]);
    assert.equal(
        state.queueModelName('mlx-community/Qwen3.5-9B-OptiQ-4bit'),
        'Qwen3.5-9B-OptiQ-4bit'
    );
});

test('queue copy leaves a repo id without an org untouched', () => {
    const state = stateWithModels([]);
    assert.equal(state.queueModelName('standalone-model'), 'standalone-model');
    assert.equal(state.queueModelName(''), '');
});
