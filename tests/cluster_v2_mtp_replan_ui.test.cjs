const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(
    path.join(__dirname, '../omlx/admin/static/js/cluster_v2.js'), 'utf8'
);
const context = {
    window: {t: key => key},
    document: {},
    localStorage: {getItem: () => null, setItem: () => {}},
    navigator: {language: 'en'},
};
const create = vm.runInNewContext(source + '\n clusterV2Wizard;', context);

function replanBody(execution) {
    const state = create();
    const deployment = {deployment_id: 'd1', target_context_tokens: 32768};
    state.configuredDeployment = () => deployment;
    state.deploymentExecution = () => execution;
    return state.executionReplanBody('throughput');
}

// A profile change on a deployment running Lightning MTP must carry mtp along,
// otherwise the replan rebuilds the deployment with MTP off.
const withMtp = replanBody({
    sampling_rank_only: false, async_overlap: false, mtp: true,
});
assert.equal(withMtp.mtp, true);
assert.equal(withMtp.execution_profile, 'throughput');

const without = replanBody({sampling_rank_only: true, async_overlap: true});
assert.equal(without.mtp, false);
console.log('cluster v2 replan carries mtp');
