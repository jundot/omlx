// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8');
const app = vm.runInNewContext(source + '\n dashboard;', {
    localStorage: { getItem: () => null }, window: { t: key => key }, console, URL,
})();
const prefix = 'Qwen3.8-Flash-Next-UD-Q4_K_XL';
const names = Array.from({ length: 4 }, (_, i) =>
    `Q4-PLE8/${prefix}-${String(i + 1).padStart(5, '0')}-of-00005.gguf`);
const table = 'Q4-PLE8/Qwen3.8-Flash-Next-ple8-table-00005-of-00005.gguf';
names.push(table);
const projector = 'mmproj-BF16.gguf';
const group = `Q4-PLE8/${prefix}-Q4-PLE8`;
app.hfModelDetail = {
    files: names.map(name => ({ name, size: 100, group, mmproj: projector }))
        .concat([{ name: projector, size: 10 }]),
};

// Picking the table fetches every body shard and the shared projector.
app.toggleHfFile(app.hfModelDetail.files[4]);
assert.deepEqual(Array.from(app.hfSelectedNames()), [...names, projector]);
assert.equal(app.hfFileSize(app.hfModelDetail.files[0]), 500);
app.toggleHfFile(app.hfModelDetail.files[0]);
assert.deepEqual(Array.from(app.hfSelectedNames()), [projector]);

// Completed downloads use the first body shard, even if the table was selected first.
const task = { repo_id: 'Vardaan97/Flash-Next-Splash-GGUF', files: [table, projector, ...names.slice(0, 4)] };
assert.equal(app.taskModelId(task), `${prefix}-Q4-PLE8`);
assert.equal(app.taskGgufFiles(task).split(', ').length, 5);
assert.equal(app.taskModelId({ files: ['Q2_0/Model-Q2_0-00001-of-00002.gguf'] }), 'Model-Q2_0');
for (const name of ['MTP/draft-00001-of-00002.gguf', '.hidden/Model-00001-of-00002.gguf', projector]) {
    assert.equal(app.isGgufModelFile(name), false);
}
