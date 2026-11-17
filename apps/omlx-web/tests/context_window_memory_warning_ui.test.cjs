// SPDX-License-Identifier: Apache-2.0
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const WARNING = 'modal.model_settings.max_context_window_memory_warning';
const context = {
    localStorage: {getItem: () => null},
    window: {t: k => (k === WARNING ? 'About {tokens} fit' : k)},
    console,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8'), context);

// The Context Window row warns when the window is longer than the prompt the
// memory guard admitted after the model loaded (#5147).
const app = context.dashboard();
const fits = `About ${(20352).toLocaleString()} fit`;
app.selectedModel = {id: 'model-a', model_context_length: 40960, memory_fit_context: 20352};
app.modelSettings.max_context_window = null;
app.globalSettings.sampling.max_context_window_policy = null;
assert.equal(app.contextWindowMemoryWarning, fits);

// The global cap lowers the default window below the fit.
app.globalSettings.sampling.max_context_window_policy = 16384;
assert.equal(app.contextWindowMemoryWarning, '');

// A per-model value wins over the cap, as on the server.
app.modelSettings.max_context_window = 131072;
assert.equal(app.contextWindowMemoryWarning, fits);
app.modelSettings.max_context_window = 20352;
assert.equal(app.contextWindowMemoryWarning, '');

// Without a native length the global fallback is the window, and the cap
// does not apply to it, as on the server.
app.selectedModel = {id: 'model-a', model_context_length: null, memory_fit_context: 20352};
app.modelSettings.max_context_window = null;
app.globalSettings.sampling.max_context_window = 32768;
app.globalSettings.sampling.max_context_window_policy = 16384;
assert.equal(app.contextWindowMemoryWarning, fits);

// Diffusion models have no context window.
app.modelSettings.is_diffusion_model = true;
assert.equal(app.contextWindowMemoryWarning, '');
app.modelSettings.is_diffusion_model = false;

// Unloaded models have no measured fit.
app.selectedModel = {id: 'model-a', model_context_length: 40960, memory_fit_context: null};
app.modelSettings.max_context_window = 131072;
assert.equal(app.contextWindowMemoryWarning, '');
