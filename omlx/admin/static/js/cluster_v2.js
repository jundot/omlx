// oMLX Cluster v2 wizard — Alpine.js component backing
// omlx/admin/templates/dashboard/_cluster_v2.html.
//
// Contract (ops/notes/omlx_cluster_v2_spec.md, Module C): consume ONLY the
// Module A/B endpoints plus the pre-existing planner/activate API:
//
//   Module A/B (cluster v2):
//     GET    /api/cluster/devices               — {paired, discovered, self}, polled at 1 Hz
//     POST   /api/cluster/pair/approve          — {node_id, code}
//     POST   /api/cluster/pair/deny             — {node_id}
//     POST   /api/cluster/pair/join             — {coordinator_addr} — joiner: mint + show the code
//     GET    /api/cluster/pair/join             — local join snapshot, polled at 1 Hz (drives approval)
//     POST   /api/cluster/pair/join/cancel      — abandon the join in progress
//     POST   /api/cluster/devices/manual        — {ip, port} seed + probe a peer by address
//     DELETE /api/cluster/devices/{node_id}     — unpair
//     GET    /api/cluster/discovery/health      — {multicast_rx_within_5s: bool,
//           last_multicast_rx_at: float|null, mdns_active: bool, transport: str}
//           Powers the "Local network permission" self-test check row.
//           Implemented at discovery_routes.py; a 404 from an older build
//           degrades the row to "skipped", never a red failure (beacon loss
//           only affects discovery, not pairing via Add by IP).
//   Existing planner/activate API (omlx/cluster/routes.py, unchanged):
//     POST   /admin/api/cluster/models          — per-node model inventory
//     POST   /admin/api/cluster/catalogue       — model fit across nodes
//     POST   /admin/api/cluster/peer-probe      — SSH reachability / versions / RDMA
//     POST   /admin/api/cluster/autoconfigure   — signed, launch-ready proposal
//     POST   /admin/api/cluster/stage           — start a resumable job copying
//           the model to plan members that lack it (auto-staging: the wizard
//           runs this as phase 1 of activation when /models shows the model is
//           not on every Mac; the body is {activation, parallel} where
//           activation is byte-identical to the /deployments body)
//     GET    /admin/api/cluster/stage/{job_id}  — per-node copy progress,
//           polled at 1 Hz by a dedicated timer (not tick(): activation must
//           survive a tab switch). 404 mid-poll = coordinator restarted, the
//           job is lost; re-POSTing /stage is safe (verified files skip).
//     GET    /admin/api/cluster/deployments     — active deployments
//     POST   /admin/api/cluster/deployments     — activate an approved plan
//     DELETE /admin/api/cluster/deployments/{id}— deactivate
//     POST   /admin/api/cluster/replan           — signed preview/apply reload
//     GET    /admin/api/cluster/tp-layout-qualifications — exact heterogeneous
//           TP evidence and store health
//
// State machine (wizardState()): empty → discovering → device_card → pairing →
// checks → plan → active, with error as an overlay state (banner + toasts,
// never a modal). N device cards are rendered; there is no 2-Mac cap.
//
// This file is loaded as a plain script before Alpine initializes, exactly
// like dashboard.js, so `clusterV2Wizard` is a global factory referenced from
// x-data in _cluster_v2.html.

function clusterV2Wizard() {
    const CLUSTER_V2_API = {
        devices: '/api/cluster/devices',
        discoveryHealth: '/api/cluster/discovery/health',
        pairApprove: '/api/cluster/pair/approve',
        pairDeny: '/api/cluster/pair/deny',
        pairJoin: '/api/cluster/pair/join',
        pairJoinCancel: '/api/cluster/pair/join/cancel',
        manualDevice: '/api/cluster/devices/manual',
        unpair: (nodeId) =>
            `/api/cluster/devices/${encodeURIComponent(nodeId)}`,
        models: '/admin/api/cluster/models',
        catalogue: '/admin/api/cluster/catalogue',
        peerProbe: '/admin/api/cluster/peer-probe',
        autoconfigure: '/admin/api/cluster/autoconfigure',
        stage: '/admin/api/cluster/stage',
        stageJob: (id) =>
            `/admin/api/cluster/stage/${encodeURIComponent(id)}`,
        nodeRoles: '/admin/api/cluster/node-roles',
        nodeBudgets: '/admin/api/cluster/node-budgets',
        runtime: '/admin/api/cluster/runtime',
        deployments: '/admin/api/cluster/deployments',
        replan: '/admin/api/cluster/replan',
        tpQualifications: '/admin/api/cluster/tp-layout-qualifications',
        deployment: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}`,
        deploymentLoad: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}/load`,
        deploymentUnload: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}/unload`,
    };

    // exo-style data layer: one snapshot endpoint polled once per second
    // while the tab is visible; the UI is a pure function of the snapshot.
    const CLUSTER_V2_POLL_MS = 1000;
    const CLUSTER_V2_DEPLOYMENTS_EVERY_TICKS = 5;
    // A missing v2 backend (404) flips the error state immediately; flaky
    // networks get a few grace failures first.
    const CLUSTER_V2_FAILURE_GRACE = 3;

    const CLUSTER_V2_LINK_META = {
        tb: { label: 'Thunderbolt', icon: 'zap' },
        ethernet: { label: 'Ethernet', icon: 'cable' },
        wifi: { label: 'Wi-Fi', icon: 'wifi' },
        // This is the address used to find/control the peer. It is not the
        // collective fabric selected by the signed deployment.
        tailscale: { label: 'Tailscale control', icon: 'globe' },
        unknown: { label: 'Network', icon: 'help-circle' },
    };

    // Offline mirror of omlx/cluster/node_role.py (NodeRole.reserve_for):
    // a workstation keeps max(32 GiB, 50%) of its Mac, a headless node keeps
    // 10%. The wizard prefers GET /admin/api/cluster/node-roles, which exposes
    // these same numbers from the server; this mirror exists only so the
    // usable-budget labels still render when that endpoint is unreachable.
    // If node_role.py changes, this mirror must change with it.
    const CLUSTER_V2_ROLE_FALLBACK = {
        workstation: {
            key: 'workstation',
            label: 'Workstation',
            reserve_bytes: 32 * 1024 ** 3,
            reserve_fraction: 0.5,
        },
        headless: {
            key: 'headless',
            label: 'Headless',
            reserve_bytes: 0,
            reserve_fraction: 0.1,
        },
    };

    // English fallbacks for the cluster.v2.* strings the wizard adds. The
    // dashboard resolves window.t against en.json-filled locale_json, so these
    // only matter when window.t is unavailable (offline component tests).
    // Keep in sync with omlx/admin/i18n/en.json.
    const CLUSTER_V2_STRINGS = {
        'cluster.v2.strategy.title': 'How the model is split',
        'cluster.v2.strategy.auto': 'Auto',
        'cluster.v2.strategy.tensor': 'Tensor',
        'cluster.v2.strategy.pipeline': 'Pipeline',
        'cluster.v2.strategy.disaggregated': 'Phase split',
        'cluster.v2.strategy.recommended': 'Recommended',
        'cluster.v2.strategy.hint.auto':
            'oMLX picks the split that fits this model and your link',
        'cluster.v2.strategy.hint.tensor':
            'Every Mac works on every token — needs a fast link',
        'cluster.v2.strategy.hint.pipeline':
            'Each Mac holds a different slice of the layers',
        'cluster.v2.strategy.hint.disaggregated':
            'One full replica prefills while the other full replica decodes',
        'cluster.v2.strategy.tensor_needs_two':
            'Tensor parallelism needs 2+ Macs',
        'cluster.v2.strategy.tensor_unsupported':
            "This model's attention heads cannot be split across Macs",
        'cluster.v2.strategy.pipeline_unsupported':
            'This model does not support pipeline stages — use Tensor instead',
        'cluster.v2.split.tensor_caption':
            'Tensor split — each Mac holds 1/{count} of every layer',
        'cluster.v2.split.tensor_share': '1/{count} of every layer',
        'cluster.v2.models.on_every_mac': 'on every Mac',
        'cluster.v2.models.partial':
            'on {have} of {total} Macs — copied at activation',
        'cluster.v2.staging.title':
            'Copying the model to your Macs — {done} of {total} ready',
        'cluster.v2.staging.waiting': 'waiting',
        'cluster.v2.staging.checking': 'checking what is already there…',
        'cluster.v2.staging.copying': 'copying {done} of {total} files',
        'cluster.v2.staging.already_there': 'already has it',
        'cluster.v2.staging.ready': 'ready',
        'cluster.v2.staging.failed': 'copy failed',
        'cluster.v2.staging.lost':
            'The coordinator restarted mid-copy — press Activate to resume.',
        'cluster.v2.staging.unsupported':
            'This build cannot copy models automatically — trying to activate directly.',
        'cluster.v2.staging.button_copying_one': 'Copying model to 1 Mac…',
        'cluster.v2.staging.button_copying': 'Copying model to {count} Macs…',
        'cluster.v2.checks.beacon_label':
            'Local network permission (discovery only)',
        'cluster.v2.steps.plan_hint_tensor': 'Every layer, on every Mac',
    };
    const t = (key) =>
        typeof window !== 'undefined' && typeof window.t === 'function'
            ? window.t(key)
            : CLUSTER_V2_STRINGS[key] || key;

    return {
        // ---- snapshot state -------------------------------------------------
        devicesPayload: null,
        devicesLoaded: false,
        devicesError: '',
        devicesFailureCount: 0,
        devicesUnreachable: false,
        deploymentsPayload: [],
        deploymentsLoaded: false,
        // Deployments are durable configuration. Runtime is the separate,
        // observed proof that their rank processes are loading or resident.
        // Keeping these snapshots separate prevents a registry record from
        // resurrecting an unloaded model after an unload or reboot.
        runtimePayload: null,
        runtimeLoaded: false,
        runtimeError: '',
        discoveryHealth: null,
        discoveryHealthUnsupported: false,

        // ---- wizard cursor ---------------------------------------------------
        // Null = derive from the snapshot. Explicit values: 'checks', 'plan'.
        stage: null,
        pairing: { target: null, code: '', busy: false, error: '' },

        // ---- joiner side (this Mac shows the code, the other Mac approves) ---
        // Server-driven snapshot from GET /api/cluster/pair/join; polled in
        // tick() so the panel survives reloads and approval completes by
        // itself. target_name is client-local context for friendly toasts.
        join: {
            state: 'idle',
            code: null,
            expires_at: null,
            coordinator_addr: null,
            seconds_remaining: 0,
            error: null,
            busy: false,
            target_name: '',
        },
        joinApprovedNotified: false,
        joinDeniedNotified: false,

        // ---- add by IP (when multicast discovery is unavailable) -------------
        manualAddr: '',
        manualBusy: false,
        manualError: '',
        checks: {
            started: false,
            running: false,
            probes: {},
            benchmark: null,
            benchmarkRunning: false,
            ranAt: null,
        },

        // ---- plan ------------------------------------------------------------
        modelOptions: [],
        modelsLoading: false,
        modelsError: '',
        selectedModelPath: '',
        modelSearch: '',
        // Execution strategy for an ordinary split or full-replica phases.
        // The server resolves its TP degree, host order and backend together;
        // the client never reconstructs those decisions after preview.
        planStrategy: 'auto',
        phasePrefillRank: 1,
        // Scheduler limits are selected as one of the server-owned execution
        // profiles. The worker may safely reduce these values for available
        // headroom, and the active view always renders those resolved values.
        // Continuous batching itself is automatic for batchable models; this
        // profile controls its target width and prompt/decode admission limits.
        executionProfile: 'balanced',
        // The volatile rank-local prompt LRU is always enabled. Persistent
        // SSD boundary snapshots are explicit because their bounded detached
        // payloads consume memory and disk even though writes run in back.
        promptCacheSsd: false,
        promptCacheSsdMaxGiB: 20,
        targetContextTokens: 32768,
        // Per-model strategy advice from POST /admin/api/cluster/catalogue
        // (null = not attempted yet). catalogueFailed switches the
        // recommendation badge to the fast-transport heuristic.
        catalogueModels: null,
        catalogueLoading: false,
        catalogueFailed: false,
        plan: null,
        // Complete /autoconfigure response. Its activation object is the only
        // payload allowed through staging and deployment.
        planProposal: null,
        planRequestRevision: 0,
        planLoading: false,
        planError: '',
        // Explicit per-node role picks (node_id → 'workstation' | 'headless').
        // Nodes without an entry keep the long-standing default: this Mac is
        // a workstation, every peer is headless.
        nodeRoles: {},
        roleOptions: [],
        // Parsed from a /plan 400: { shortfallBytes, canFixWithHeadless }.
        planFitFailure: null,
        activateBusy: false,
        // ---- auto-staging (phase 1 of activation) ----------------------------
        // When /models shows the selected model is not on every plan Mac,
        // activatePlan first POSTs /stage and polls the job at 1 Hz; on
        // completion the same request body goes to /deployments.
        stagingJob: null, // last /stage snapshot; null = idle
        stagingActivation: null, // exact server proposal frozen for this job
        stagingError: '', // job-level or POST-level failure, shown inline
        // Dedicated 1 Hz poller — NOT tick(), which early-returns when the
        // tab is hidden; activation must survive a tab switch.
        stagingTimer: null,
        confirmUnpairFor: '',
        confirmDeactivateFor: '',
        confirmUnloadFor: '',
        confirmChangeModelFor: '',
        clusterLifecycleBusy: false,
        executionReplan: null,
        executionReplanBusy: false,
        // Pairing is durable and model-independent. Adding a paired Mac to
        // resident ranks is a separate signed activation preview.
        membershipPanelOpen: false,
        membershipBusy: false,
        membershipError: '',
        membershipProposal: null,
        membershipReturnStage: null,
        tpQualifications: null,
        tpQualificationsError: '',

        // ---- feedback ----------------------------------------------------------
        toasts: [],
        toastSeq: 0,
        installCommandCopied: false,

        pollTimer: null,
        tickCount: 0,

        // =====================================================================
        // Lifecycle
        // =====================================================================
        init() {
            this.tick();
            this.refreshTpQualifications();
            this.pollTimer = setInterval(() => this.tick(), CLUSTER_V2_POLL_MS);
        },

        wizardVisible() {
            // mainTab and clusterLegacyView live on ancestor scopes
            // (dashboard() and the dashboard.html wrapper respectively).
            return (
                this.mainTab === 'cluster' &&
                !this.clusterLegacyView &&
                !document.hidden
            );
        },

        async tick() {
            if (!this.wizardVisible()) return;
            await this.refreshDevices();
            await this.refreshJoinState();
            await this.refreshRuntime();
            this.tickCount += 1;
            if (
                !this.deploymentsLoaded ||
                this.tickCount % CLUSTER_V2_DEPLOYMENTS_EVERY_TICKS === 0
            ) {
                await this.refreshDeployments();
            }
        },

        // =====================================================================
        // API helpers
        // =====================================================================
        async apiFetch(url, options = {}) {
            const response = await fetch(url, {
                headers: { 'Content-Type': 'application/json' },
                ...options,
            });
            if (response.status === 401) {
                window.location.href = '/admin';
                throw new Error('Sign-in required');
            }
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) {
                        detail =
                            typeof payload.detail === 'string'
                                ? payload.detail
                                : JSON.stringify(payload.detail);
                    }
                } catch (ignored) {
                    /* non-JSON error body */
                }
                const error = new Error(detail);
                error.status = response.status;
                throw error;
            }
            if (response.status === 204) return null;
            return response.json();
        },

        async refreshDevices() {
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.devices);
                this.devicesPayload = payload || {};
                this.devicesLoaded = true;
                this.devicesError = '';
                this.devicesFailureCount = 0;
                this.devicesUnreachable = false;
            } catch (error) {
                this.devicesFailureCount += 1;
                this.devicesError = error?.message || 'Cluster API unreachable';
                // 404 means the v2 discovery backend is not serving — that is
                // a hard, actionable failure, not network noise.
                if (
                    error?.status === 404 ||
                    this.devicesFailureCount >= CLUSTER_V2_FAILURE_GRACE
                ) {
                    this.devicesUnreachable = true;
                }
            }
        },

        async refreshDeployments() {
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.deployments);
                this.deploymentsPayload = payload?.deployments || [];
                this.deploymentsLoaded = true;
                this.ensureColdModelSelection();
            } catch (error) {
                // Deployments are the pre-existing API; a failure here should
                // not tear down the discovery UI, just surface a toast once.
                if (this.deploymentsLoaded) {
                    this.notify(
                        'warning',
                        error?.message || 'Could not refresh deployments',
                    );
                }
            }
        },

        async refreshRuntime() {
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.runtime);
                this.runtimePayload = payload || { jobs: [], launchers: [] };
                this.runtimeLoaded = true;
                this.runtimeError = '';
                this.ensureColdModelSelection();
            } catch (error) {
                // Fail closed. A previous ready snapshot must never remain green
                // when the ownership endpoint can no longer prove residency.
                this.runtimePayload = null;
                this.runtimeLoaded = false;
                this.runtimeError =
                    error?.message || 'Cluster runtime status is unavailable';
            }
        },

        async refreshTpQualifications() {
            try {
                this.tpQualifications = await this.apiFetch(
                    CLUSTER_V2_API.tpQualifications,
                );
                this.tpQualificationsError = '';
            } catch (error) {
                this.tpQualifications = null;
                this.tpQualificationsError =
                    error?.message || 'TP qualification evidence is unavailable';
            }
        },

        async refreshDiscoveryHealth() {
            try {
                this.discoveryHealth = await this.apiFetch(
                    CLUSTER_V2_API.discoveryHealth,
                );
                this.discoveryHealthUnsupported = false;
            } catch (error) {
                if (error?.status === 404) {
                    // Older builds predate the endpoint (implemented at
                    // discovery_routes.py); the check row degrades to
                    // "skipped" instead of a misleading red failure.
                    this.discoveryHealthUnsupported = true;
                    this.discoveryHealth = null;
                } else {
                    this.discoveryHealth = null;
                }
            }
        },

        // The joiner snapshot is server-owned, so a page reload mid-join
        // restores the panel and the coordinator's approval completes on the
        // next tick without any user action.
        async refreshJoinState() {
            try {
                const snapshot = await this.apiFetch(CLUSTER_V2_API.pairJoin);
                if (!snapshot) return;
                const previous = this.join.state;
                this.join = { ...this.join, ...snapshot, busy: false };
                if (
                    snapshot.state === 'approved' &&
                    !this.joinApprovedNotified
                ) {
                    this.joinApprovedNotified = true;
                    this.notify(
                        'success',
                        `This Mac joined ${this.joinTargetName()}'s cluster.`,
                    );
                    await this.refreshDevices();
                    this.startChecks();
                } else if (
                    snapshot.state === 'denied' &&
                    previous !== 'denied' &&
                    !this.joinDeniedNotified
                ) {
                    this.joinDeniedNotified = true;
                    this.notify(
                        'error',
                        `${this.joinTargetName()} denied the join request.`,
                    );
                    // Denied is terminal server-side; reset locally so the
                    // panel clears instead of sticking on the refusal.
                    await this.cancelJoin({ silent: true });
                }
            } catch (error) {
                // A 404 means this backend predates the joiner endpoints —
                // stay idle rather than tearing down the rest of the wizard.
                if (error?.status !== 404) {
                    this.join = {
                        ...this.join,
                        error: error?.message || 'Join status unavailable',
                    };
                }
            }
        },

        // =====================================================================
        // Snapshot selectors
        // =====================================================================
        selfDevice() {
            return this.devicesPayload?.self || null;
        },

        pairedDevices() {
            const paired = this.devicesPayload?.paired;
            return Array.isArray(paired) ? paired : [];
        },

        discoveredDevices() {
            const discovered = this.devicesPayload?.discovered;
            if (!Array.isArray(discovered)) return [];
            return discovered.filter(
                (device) =>
                    device && !device.paired && device.state !== 'awaiting_approval',
            );
        },

        pendingApprovals() {
            const discovered = this.devicesPayload?.discovered;
            if (!Array.isArray(discovered)) return [];
            return discovered.filter(
                (device) => device && device.state === 'awaiting_approval',
            );
        },

        allDevices() {
            // N device cards — the v1 GUI's 2-Mac cap is gone.
            const devices = [];
            const self = this.selfDevice();
            if (self) devices.push({ ...self, is_self: true, paired: true });
            for (const device of this.pairedDevices()) devices.push(device);
            for (const device of this.discoveredDevices()) devices.push(device);
            for (const device of this.pendingApprovals()) devices.push(device);
            return devices;
        },

        configuredDeployment() {
            const deployments = Array.isArray(this.deploymentsPayload)
                ? this.deploymentsPayload
                : [];
            if (!deployments.length) return null;

            // A registry may retain several signed setups (for example, a
            // cold DS4 TP plan plus the currently loaded Qwen phase plan).
            // The active card must follow runtime ownership, not registry
            // insertion order, or it presents an unloaded model as running.
            const jobs = Array.isArray(this.runtimePayload?.jobs)
                ? this.runtimePayload.jobs
                : [];
            const launchers = Array.isArray(this.runtimePayload?.launchers)
                ? this.runtimePayload.launchers
                : [];
            const deploymentForId = (id) =>
                deployments.find((deployment) => deployment?.deployment_id === id);
            const selectJob = (predicate) => {
                const job = jobs.find(
                    (candidate) => candidate?.deployment_id && predicate(candidate),
                );
                return job ? deploymentForId(job.deployment_id) : null;
            };
            const loaded =
                selectJob(
                    (job) =>
                        job.ownership === 'loaded' &&
                        job.live === true &&
                        job.phase === 'ready',
                ) || selectJob((job) => job.ownership === 'loaded');
            if (loaded) return loaded;
            const loading =
                selectJob((job) => job.ownership === 'loading') ||
                (() => {
                    const launcher = launchers.find(
                        (candidate) =>
                            candidate?.deployment_id &&
                            ['preflight', 'loading'].includes(candidate.phase),
                    );
                    return launcher
                        ? deploymentForId(launcher.deployment_id)
                        : null;
                })();
            // Once runtime ownership is known, a durable-but-cold setup is not
            // an active deployment. Present the model picker instead of
            // arbitrarily choosing the first registry row (which may be an old
            // DS4 plan while Qwen was the model the user just unloaded).
            return loading || (this.runtimeLoaded ? null : deployments[0]);
        },

        ensureColdModelSelection() {
            if (
                !this.runtimeLoaded ||
                !this.deploymentsLoaded ||
                !this.deploymentsPayload.length ||
                this.configuredDeployment() ||
                this.stage === 'plan' ||
                !this.pairedDevices().length
            ) {
                return;
            }
            this.resetModelPicker();
            this.enterPlan();
        },

        deploymentRuntimeJobs(deployment = this.configuredDeployment()) {
            const id = deployment?.deployment_id;
            const jobs = this.runtimePayload?.jobs;
            if (!id || !Array.isArray(jobs)) return [];
            return jobs.filter((job) => job?.deployment_id === id);
        },

        deploymentRuntimeJob(deployment = this.configuredDeployment()) {
            const jobs = this.deploymentRuntimeJobs(deployment);
            return (
                jobs.find(
                    (job) => Number(job.rank) === 0 && job.live === true,
                ) ||
                jobs.find((job) => job.live === true) ||
                jobs.find((job) => Number(job.rank) === 0) ||
                jobs[0] ||
                null
            );
        },

        deploymentMetricsJob(deployment = this.configuredDeployment()) {
            const jobs = this.deploymentRuntimeJobs(deployment).filter(
                (job) => job?.metrics && typeof job.metrics === 'object',
            );
            return (
                jobs.find(
                    (job) => Number(job.rank) === 0 && job.live === true,
                ) ||
                jobs.find((job) => job.live === true) ||
                jobs.find((job) => Number(job.rank) === 0) ||
                jobs[0] ||
                null
            );
        },

        deploymentExecution(deployment = this.configuredDeployment()) {
            const job =
                this.deploymentMetricsJob(deployment) ||
                this.deploymentRuntimeJob(deployment);
            const metricsExecution = job?.metrics?.execution;
            if (metricsExecution && typeof metricsExecution === 'object') {
                return metricsExecution;
            }
            if (job?.execution && typeof job.execution === 'object') {
                return job.execution;
            }
            const configured = deployment?.execution;
            return configured && typeof configured === 'object'
                ? configured
                : null;
        },

        deploymentExecutionProfileLabel(
            deployment = this.configuredDeployment(),
        ) {
            const profile = String(
                this.deploymentExecution(deployment)?.profile || 'balanced',
            );
            return profile.charAt(0).toUpperCase() + profile.slice(1);
        },

        deploymentBatchStatus(deployment = this.configuredDeployment()) {
            const job = this.deploymentRuntimeJob(deployment);
            const metrics = job?.metrics || {};
            const pipeline = metrics.pipeline || {};
            const execution = this.deploymentExecution(deployment) || {};
            const capability = job?.optimizations?.coalesced_batching;
            const target = Math.max(
                1,
                Number(
                    pipeline.microbatch_target ||
                        execution.pipeline_microbatch_size ||
                        1,
                ),
            );
            const lastSize = Math.max(
                0,
                Number(pipeline.last_batch?.coalesced_batch_size || 0),
            );
            if (capability?.active === false) {
                return {
                    label: 'Unavailable',
                    detail:
                        capability.reason ||
                        'This model cannot merge its request caches.',
                    tone: 'bg-amber-50 border-amber-200 text-amber-700',
                    target,
                };
            }
            if (target <= 1 || capability?.enabled === false) {
                return {
                    label: 'Sequential',
                    detail: 'The resolved batch target is one request.',
                    tone: 'bg-neutral-50 border-neutral-200 text-neutral-600',
                    target,
                };
            }
            if (lastSize > 1) {
                return {
                    label: `Batched ${lastSize}`,
                    detail: `Last scheduler step coalesced ${lastSize} of ${target} possible requests.`,
                    tone: 'bg-green-50 border-green-200 text-green-700',
                    target,
                };
            }
            return {
                label: 'Automatic',
                detail: `Enabled for overlapping compatible requests, up to ${target} per scheduler step.`,
                tone: 'bg-green-50 border-green-200 text-green-700',
                target,
            };
        },

        deploymentRequestMetrics(deployment = this.configuredDeployment()) {
            const metrics = this.deploymentMetricsJob(deployment)?.metrics;
            if (!metrics) return [];
            const active = Array.isArray(metrics.active_request_metrics)
                ? metrics.active_request_metrics.filter(
                    (request) => request && request.status === 'running',
                )
                : [];
            if (active.length) {
                return active.map((request) => ({ ...request, _history: false }));
            }
            const last = metrics.last_request;
            if (!last || typeof last !== 'object') return [];
            return [{
                ...last,
                // Old rank markers expose only last_request.  While one is
                // active it is still live; otherwise keep the completed row
                // visible so a fast request is not missed between 1 Hz polls.
                _history: Number(metrics.active_requests || 0) === 0,
            }];
        },

        deploymentRequestCountLabel(deployment = this.configuredDeployment()) {
            const metrics = this.deploymentMetricsJob(deployment)?.metrics;
            const active = Math.max(0, Number(metrics?.active_requests || 0));
            const hidden = Math.max(
                0,
                Number(metrics?.active_request_metrics_truncated || 0),
            );
            if (active > 0) {
                return `${active} active${hidden ? ` · ${hidden} not shown` : ''}`;
            }
            return metrics?.last_request ? 'Last completed request' : 'Waiting';
        },

        requestPhaseLabel(request) {
            if (request?._history) {
                return request?.status === 'failed' ? 'Failed' : 'Complete';
            }
            if (request?.prefill_progress?.active) return 'Prefill';
            if (Number(request?.completion_tokens || 0) > 0) return 'Decode';
            return 'Queued';
        },

        requestPhaseTone(request) {
            const phase = this.requestPhaseLabel(request);
            if (phase === 'Prefill') return 'bg-blue-50 border-blue-200 text-blue-700';
            if (phase === 'Decode') return 'bg-green-50 border-green-200 text-green-700';
            if (phase === 'Failed') return 'bg-red-50 border-red-200 text-red-700';
            return 'bg-neutral-50 border-neutral-200 text-neutral-600';
        },

        formatRequestRate(rate) {
            const value = Number(rate);
            if (!Number.isFinite(value) || value <= 0) return '—';
            return `${value.toFixed(value >= 100 ? 0 : 1)} tok/s`;
        },

        requestPrefillRate(request) {
            const progress = request?.prefill_progress;
            const prompt = Math.max(0, Number(request?.prompt_tokens || 0));
            const cached = Math.min(
                prompt,
                Math.max(0, Number(request?.cached_tokens || 0)),
            );
            const uncached = Math.max(0, prompt - cached);
            if (
                !progress?.active &&
                cached > 0 &&
                uncached <= Math.max(16, Math.floor(prompt * 0.01))
            ) {
                return 'Cache hit';
            }
            // The final prompt-progress callback freezes at the compute
            // boundary. Prefer it even after decode begins: TTFT also includes
            // queueing and phase handoff and can make a 900 tok/s prefill look
            // like an 11 tok/s request on a long/contended run.
            const rate = Number(
                progress?.average_speed || request?.prefill_tps || 0,
            );
            return this.formatRequestRate(rate);
        },

        deploymentPhaseMetrics(deployment = this.configuredDeployment()) {
            if (deployment?.serving_mode !== 'disaggregated') return null;
            const phase = this.deploymentMetricsJob(deployment)?.metrics?.phase_split;
            return phase && typeof phase === 'object' ? phase : null;
        },

        phaseHandoffRateLabel(deployment = this.configuredDeployment()) {
            const value = Number(
                this.deploymentPhaseMetrics(deployment)
                    ?.last_handoff_bytes_per_second || 0,
            );
            if (!(value > 0)) return 'Waiting';
            return `${(value / 1e9).toFixed(2)} GB/s`;
        },

        phaseHandoffDetail(deployment = this.configuredDeployment()) {
            const phase = this.deploymentPhaseMetrics(deployment);
            if (!phase) return 'No cache handoff measured yet';
            const gib = Number(phase.last_handoff_bytes || 0) / (1024 ** 3);
            const ms = Number(phase.last_handoff_seconds || 0) * 1000;
            return `${gib.toFixed(2)} GiB · ${ms.toFixed(0)} ms · ${Number(
                phase.last_handoff_arrays || 0,
            )} tensors`;
        },

        activeDevicePhaseRole(device, deployment = this.configuredDeployment()) {
            if (deployment?.serving_mode !== 'disaggregated') return '';
            const assignment = (deployment.assignments || []).find(
                (item) => item?.node_id === device?.node_id,
            );
            if (!assignment) return '';
            if (Number(assignment.rank) === Number(deployment.prefill_rank)) {
                return 'Prefill · full replica';
            }
            if (Number(assignment.rank) === Number(deployment.decode_rank)) {
                return 'Decode + API · full replica';
            }
            return 'Full replica';
        },

        requestPrefillDetail(request) {
            if (request?.prefill_progress?.active) return 'live average';
            const prompt = Math.max(0, Number(request?.prompt_tokens || 0));
            const cached = Math.min(
                prompt,
                Math.max(0, Number(request?.cached_tokens || 0)),
            );
            if (cached > 0) {
                return `${cached.toLocaleString()} reused · ${Math.max(
                    0,
                    prompt - cached,
                ).toLocaleString()} new`;
            }
            return 'prompt average';
        },

        requestDecodeRate(request) {
            const tokens = Math.max(0, Number(request?.completion_tokens || 0));
            if (!request?._history && tokens === 1) return 'Measuring…';
            return this.formatRequestRate(request?.decode_tps);
        },

        requestPromptDetail(request) {
            const prompt = Math.max(0, Number(request?.prompt_tokens || 0));
            const cached = Math.min(
                prompt,
                Math.max(0, Number(request?.cached_tokens || 0)),
            );
            const progress = request?.prefill_progress;
            if (progress?.active) {
                const processed = Math.max(0, Number(progress.processed || 0));
                const total = Math.max(0, Number(progress.total || 0));
                return `${processed.toLocaleString()} / ${total.toLocaleString()} new`;
            }
            const uncached = Math.max(0, prompt - cached);
            return `${uncached.toLocaleString()} new${cached ? ` · ${cached.toLocaleString()} cached` : ''}`;
        },

        requestDecodeDetail(request) {
            const tokens = Math.max(0, Number(request?.completion_tokens || 0));
            return `${tokens.toLocaleString()} generated`;
        },

        deploymentRuntimeLauncher(deployment = this.configuredDeployment()) {
            const id = deployment?.deployment_id;
            const launchers = this.runtimePayload?.launchers;
            if (!id || !Array.isArray(launchers)) return null;
            return (
                launchers.find((launcher) => launcher?.deployment_id === id) ||
                null
            );
        },

        deploymentRuntimeState(deployment = this.configuredDeployment()) {
            if (!deployment) return 'none';
            if (!this.runtimeLoaded) return 'unknown';

            const jobs = this.deploymentRuntimeJobs(deployment);
            const launcher = this.deploymentRuntimeLauncher(deployment);
            const terminalPhases = new Set([
                'failed',
                'peer_lost',
                'launcher_lost',
            ]);

            // Runtime ownership is reconciled server-side against the engine
            // pool. Prefer that current ownership before retained diagnostic
            // markers: a failed marker intentionally survives for support,
            // and is briefly relabelled `loading` when a retry starts before
            // the new rank overwrites it. Letting its old phase/error win made
            // the card flash Failed -> Failed -> Ready and hid the real load
            // progress. A live ready marker or an active loading owner is
            // authoritative; terminal evidence wins only when neither exists.
            if (
                jobs.some(
                    (job) =>
                        job?.ownership === 'loaded' &&
                        job?.live === true &&
                        job?.phase === 'ready',
                )
            ) {
                return 'ready';
            }
            if (
                jobs.some(
                    (job) =>
                        job?.ownership === 'loading',
                ) ||
                ['preflight', 'loading'].includes(launcher?.phase)
            ) {
                return 'loading';
            }
            if (
                jobs.some(
                    (job) =>
                        terminalPhases.has(job?.phase) ||
                        (typeof job?.error === 'string' && !!job.error.trim()),
                ) ||
                terminalPhases.has(launcher?.phase) ||
                launcher?.returncode != null ||
                (typeof launcher?.failure_reason === 'string' &&
                    !!launcher.failure_reason.trim())
            ) {
                return 'failed';
            }
            if (jobs.some((job) => job?.ownership === 'loaded')) {
                // The pool still claims ownership but there is no fresh ready
                // marker. Present this as a failure, never as a loaded model.
                return 'failed';
            }
            return 'configured';
        },

        activeDeployment() {
            const deployment = this.configuredDeployment();
            return this.deploymentRuntimeState(deployment) === 'ready'
                ? deployment
                : null;
        },

        deploymentFailureReason(deployment = this.configuredDeployment()) {
            const job = this.deploymentRuntimeJobs(deployment).find(
                (item) =>
                    ['failed', 'peer_lost', 'launcher_lost'].includes(
                        item?.phase,
                    ) || item?.error,
            );
            if (job?.error) return String(job.error);
            const launcher = this.deploymentRuntimeLauncher(deployment);
            if (launcher?.failure_reason) {
                return String(launcher.failure_reason);
            }
            const tail = Array.isArray(launcher?.stderr_tail)
                ? launcher.stderr_tail.filter(Boolean).slice(-1)[0]
                : '';
            return tail
                ? String(tail)
                : 'The worker stopped reporting a live ready state.';
        },

        deploymentStatus(deployment = this.configuredDeployment()) {
            const state = this.deploymentRuntimeState(deployment);
            if (state === 'ready') {
                return {
                    state,
                    eyebrow: 'Cluster running',
                    label: 'Ready',
                    detail:
                        'The distributed weights are resident and available through oMLX.',
                    tone: 'bg-green-50 border-green-200 text-green-700',
                    pulse: true,
                };
            }
            if (state === 'loading') {
                return {
                    state,
                    eyebrow: 'Starting cluster',
                    label: 'Loading',
                    detail:
                        'oMLX is loading and validating the model across your Macs.',
                    tone: 'bg-blue-50 border-blue-200 text-blue-700',
                    pulse: true,
                };
            }
            if (state === 'failed') {
                return {
                    state,
                    eyebrow: 'Cluster needs attention',
                    label: 'Failed',
                    detail: this.deploymentFailureReason(deployment),
                    tone: 'bg-red-50 border-red-200 text-red-700',
                    pulse: false,
                };
            }
            if (state === 'unknown') {
                return {
                    state,
                    eyebrow: 'Cluster configured',
                    label: 'Checking',
                    detail: this.runtimeError
                        ? `Runtime status unavailable: ${this.runtimeError}`
                        : 'Checking whether the distributed weights are resident.',
                    tone: 'bg-neutral-50 border-neutral-200 text-neutral-600',
                    pulse: true,
                };
            }
            return {
                state: 'configured',
                eyebrow: 'Cluster configured',
                label: 'Not loaded',
                detail:
                    'The placement is saved, but no model weights are resident. This is expected after an unload or reboot.',
                tone: 'bg-amber-50 border-amber-200 text-amber-700',
                pulse: false,
            };
        },

        deploymentCacheLabel(deployment = this.configuredDeployment()) {
            const enabled = Boolean(
                deployment?.execution?.prompt_cache_ssd ??
                    deployment?.prompt_cache_ssd,
            );
            return enabled
                ? 'Prompt reuse · memory + persistent SSD snapshots · ' +
                    `${Math.round(Number(deployment?.execution?.prompt_cache_ssd_max_bytes || 20 * (1024 ** 3)) / (1024 ** 3))} GiB per rank`
                : 'Prompt reuse · memory only (SSD snapshots off)';
        },

        // =====================================================================
        // State machine — empty / discovering / device_card / pairing / checks /
        // plan / active / error
        // =====================================================================
        wizardState() {
            if (this.devicesUnreachable) return 'error';
            // A durable deployment keeps its management panel mounted, but its
            // badge is driven by deploymentRuntimeState(), not by persistence.
            if (this.configuredDeployment()) return 'active';
            if (this.stage === 'plan' && this.pairedDevices().length) {
                return 'plan';
            }
            if (this.stage === 'checks' && this.pairedDevices().length) {
                return 'checks';
            }
            if (
                this.pairing.target ||
                this.pendingApprovals().length ||
                this.joinActive()
            ) {
                return 'pairing';
            }
            if (
                this.discoveredDevices().length ||
                this.pairedDevices().length
            ) {
                return 'device_card';
            }
            // Identity exists but nobody else is on the network yet — and the
            // pre-first-load skeleton also lands here (it animates either way).
            if (!this.devicesLoaded || this.selfDevice()) return 'discovering';
            return 'empty';
        },

        wizardSteps() {
            // Snapshot the derived state once. Runtime polling can land between
            // Alpine expressions; recomputing it five times allowed one render
            // to mix Plan and Active semantics during activation.
            const wizard = this.wizardState();
            const steps = [
                { key: 'discover', title: 'Find devices', hint: 'Automatic on your network' },
                { key: 'pair', title: 'Pair', hint: 'One code, both Macs' },
                { key: 'checks', title: 'Check', hint: 'SSH · model · RDMA' },
                { key: 'plan', title: 'Split the model', hint: this.planStepHint() },
                { key: 'active', title: 'Activate', hint: 'Run across the pool' },
            ];
            // Map the 8 UI states onto the 5 step slots.
            const slotFor = {
                empty: 0,
                discovering: 0,
                device_card: 1,
                pairing: 1,
                checks: 2,
                plan: 3,
                active: 4,
            };
            const activeSlot = slotFor[wizard] ?? 0;
            const activationComplete =
                wizard === 'active' && this.deploymentRuntimeState() === 'ready';
            return steps.map((step, index) => ({
                ...step,
                state:
                    wizard === 'error'
                        ? 'todo'
                        : index < activeSlot ||
                          (activationComplete && index === 4)
                        ? 'done'
                        : index === activeSlot
                        ? 'active'
                        : 'todo',
            }));
        },

        activationDeploymentId() {
            return (
                this.configuredDeployment()?.deployment_id ||
                this.stagingActivation?.deployment_id ||
                this.activationRequestBody()?.deployment_id ||
                ''
            );
        },

        activationRuntimeJobs() {
            const jobs = Array.isArray(this.runtimePayload?.jobs)
                ? this.runtimePayload.jobs
                : [];
            const id = this.activationDeploymentId();
            if (id) return jobs.filter((job) => job?.deployment_id === id);
            return jobs.filter((job) =>
                ['loading', 'loaded'].includes(job?.ownership),
            );
        },

        activationProgressVisible() {
            return (
                this.activateBusy ||
                this.deploymentRuntimeState() === 'loading'
            );
        },

        activationLoadStage() {
            if (this.stagingJob && this.stagingJob.status !== 'completed') {
                return 'staging';
            }
            const order = [
                'initializing',
                'initializing_full_replica',
                'loading_weights',
                'materializing_fixed',
                'materializing_layers',
                'tensor_ready',
                'weights_resident',
                'validating',
                'warming_prefill_shape',
                'ready',
            ];
            const jobs = this.activationRuntimeJobs();
            if (jobs.length) {
                return jobs.reduce((slowest, job) => {
                    const candidate = String(job?.load_stage || 'initializing');
                    return order.indexOf(candidate) < order.indexOf(slowest)
                        ? candidate
                        : slowest;
                }, 'ready');
            }
            const launcher = this.deploymentRuntimeLauncher();
            if (launcher?.phase === 'preflight') return 'preflight';
            if (launcher?.phase === 'loading') return 'initializing';
            return this.activateBusy ? 'starting' : 'ready';
        },

        activationProgressPercent() {
            if (this.activationLoadStage() === 'staging') {
                const nodes = this.stagingNodes();
                const total = nodes.reduce(
                    (sum, node) => sum + Math.max(0, Number(node.files_total || 0)),
                    0,
                );
                const done = nodes.reduce(
                    (sum, node) =>
                        sum + Math.max(0, Number(node.files_completed || 0)),
                    0,
                );
                return Math.round(5 + 35 * (total > 0 ? done / total : 0));
            }
            const milestones = {
                starting: 5,
                preflight: 10,
                initializing: 15,
                initializing_full_replica: 20,
                loading_weights: 35,
                materializing_fixed: 45,
                materializing_layers: 60,
                tensor_ready: 70,
                weights_resident: 75,
                validating: 85,
                warming_prefill_shape: 94,
                ready: 100,
            };
            return milestones[this.activationLoadStage()] ?? 5;
        },

        activationProgressLabel() {
            const labels = {
                staging: 'Syncing model files across the pool',
                starting: 'Starting cluster activation',
                preflight: 'Verifying the signed deployment',
                initializing: 'Starting rank processes',
                initializing_full_replica: 'Initializing full model replicas',
                loading_weights: 'Loading model weights',
                materializing_fixed: 'Materializing shared weights',
                materializing_layers: 'Materializing model layers',
                tensor_ready: 'Finalizing tensor shards',
                weights_resident: 'Model weights are resident',
                validating: 'Validating every rank',
                warming_prefill_shape: 'Warming the prefill path',
                ready: 'Cluster ready',
            };
            const stage = this.activationLoadStage();
            return labels[stage] || labels.starting;
        },

        activationProgressDetail() {
            if (this.activationLoadStage() === 'staging') {
                return this.stagingOverallLabel();
            }
            const jobs = this.activationRuntimeJobs();
            if (!jobs.length) {
                return 'Preparing the launcher and waiting for rank heartbeats.';
            }
            const ready = jobs.filter(
                (job) => job?.phase === 'ready' && job?.live === true,
            ).length;
            const expected = Math.max(
                jobs.length,
                ...jobs.map((job) => Math.max(0, Number(job?.world_size || 0))),
            );
            return `${ready} of ${expected} ranks ready · readiness canary runs last`;
        },

        // Step-4 hint is strategy-aware: a tensor split gives every Mac every
        // layer, so "Layers per Mac" would describe the wrong thing.
        planStepHint() {
            return this.planStrategy === 'tensor'
                ? t('cluster.v2.steps.plan_hint_tensor')
                : 'Layers per Mac';
        },

        // =====================================================================
        // Version parity — actionable banner, device stays visible
        // =====================================================================
        versionMismatches() {
            const self = this.selfDevice();
            if (!self || !self.version) return [];
            return this.allDevices()
                .filter(
                    (device) =>
                        !device.is_self &&
                        device.version &&
                        device.version !== self.version,
                )
                .map((device) => ({
                    name: this.deviceName(device),
                    peerVersion: device.version,
                    selfVersion: self.version,
                }));
        },

        // =====================================================================
        // Device card helpers
        // =====================================================================
        deviceName(device) {
            return (
                device?.friendly_name ||
                device?.node_id?.slice(0, 8) ||
                'Unknown device'
            );
        },

        deviceRamGb(device) {
            const ram = device?.caps?.ram_gb;
            return typeof ram === 'number' && ram > 0 ? ram : null;
        },

        deviceRamLabel(device) {
            const ram = this.deviceRamGb(device);
            return ram ? `${ram} GB` : 'Memory unknown';
        },

        deviceChipLabel(device) {
            return device?.caps?.chip || 'Apple silicon';
        },

        deviceLinkMeta(device) {
            return (
                CLUSTER_V2_LINK_META[device?.link] ||
                CLUSTER_V2_LINK_META.unknown
            );
        },

        deviceStateTone(device) {
            if (device?.state === 'dead') {
                return 'bg-red-50 border-red-200 text-red-700';
            }
            if (device?.state === 'suspect') {
                return 'bg-amber-50 border-amber-200 text-amber-700';
            }
            return 'bg-green-50 border-green-200 text-green-700';
        },

        deviceStateLabel(device) {
            if (device?.is_self) return 'This Mac';
            if (device?.state === 'awaiting_approval') return 'Wants to join';
            if (device?.paired) return 'Paired';
            if (device?.state === 'dead') return 'Unreachable';
            if (device?.state === 'suspect') return 'Connection shaky';
            return 'Found nearby';
        },

        combinedMemoryLabel() {
            const total = this.allDevices().reduce(
                (sum, device) => sum + (this.deviceRamGb(device) || 0),
                0,
            );
            return total ? `${total} GB combined` : '';
        },

        // =====================================================================
        // Pairing (Module B)
        // =====================================================================
        beginPairing(device) {
            this.pairing = {
                target: device,
                code: '',
                busy: false,
                error: '',
            };
        },

        cancelPairing() {
            this.pairing = { target: null, code: '', busy: false, error: '' };
        },

        async submitPairApproval(device) {
            const code = (this.pairing.code || '').trim();
            if (!device || this.pairing.busy) return;
            if (!/^\d{6}$/.test(code)) {
                this.pairing.error = 'The code is the 6 digits shown on the other Mac.';
                return;
            }
            this.pairing.busy = true;
            this.pairing.error = '';
            try {
                await this.apiFetch(CLUSTER_V2_API.pairApprove, {
                    method: 'POST',
                    body: JSON.stringify({ node_id: device.node_id, code }),
                });
                this.notify(
                    'success',
                    `${this.deviceName(device)} joined the cluster.`,
                );
                this.cancelPairing();
                if (this.membershipPanelOpen) {
                    await this.refreshDevices();
                    const paired = this.pairedDevices().find(
                        (item) => item.node_id === device.node_id,
                    );
                    if (paired) await this.probePeer(paired);
                    this.membershipProposal = null;
                    this.membershipError = '';
                    if (!this.configuredDeployment()) {
                        this.membershipPanelOpen = false;
                        this.stage = this.selectedModelPath ? 'plan' : 'checks';
                        if (this.selectedModelPath) {
                            await this.runPlan();
                        } else {
                            this.startChecks();
                        }
                    }
                } else {
                    this.startChecks();
                }
            } catch (error) {
                if (error?.status === 404 || error?.status === 409) {
                    this.pairing.error =
                        'No join request from this Mac yet. On the other Mac, open its oMLX dashboard and press Pair first — then type its code here.';
                } else if (error?.status === 403 || error?.status === 429) {
                    this.pairing.error =
                        error.message ||
                        'Wrong code too many times — wait for the lockout to lift and try a fresh code.';
                } else {
                    this.pairing.error =
                        error?.message || 'Pairing failed. Try again.';
                }
            } finally {
                this.pairing.busy = false;
            }
        },

        async submitPairDenial(device) {
            if (!device) return;
            try {
                await this.apiFetch(CLUSTER_V2_API.pairDeny, {
                    method: 'POST',
                    body: JSON.stringify({ node_id: device.node_id }),
                });
                this.notify(
                    'info',
                    `Join request from ${this.deviceName(device)} denied.`,
                );
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not deny the join request',
                );
            }
            if (this.pairing.target?.node_id === device?.node_id) {
                this.cancelPairing();
            }
        },

        async unpairDevice(device) {
            if (!device) return;
            // Two-step confirm — destructive, but never a modal.
            if (this.confirmUnpairFor !== device.node_id) {
                this.confirmUnpairFor = device.node_id;
                return;
            }
            this.confirmUnpairFor = '';
            try {
                await this.apiFetch(CLUSTER_V2_API.unpair(device.node_id), {
                    method: 'DELETE',
                });
                this.notify(
                    'info',
                    `${this.deviceName(device)} was removed from the cluster.`,
                );
                await this.refreshDevices();
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not unpair this device',
                );
            }
        },

        // =====================================================================
        // Joiner side — THIS Mac shows the code; the other Mac approves it.
        // =====================================================================
        joinActive() {
            // Anything but idle keeps the joiner panel up; approved/denied
            // clear themselves through refreshJoinState().
            return !!this.join.state && this.join.state !== 'idle';
        },

        joinTargetName() {
            return (
                this.join.target_name ||
                this.join.coordinator_name ||
                this.join.coordinator_addr ||
                'the other Mac'
            );
        },

        joinCountdownLabel() {
            const total = Math.max(0, Math.round(this.join.seconds_remaining || 0));
            const minutes = Math.floor(total / 60);
            const seconds = String(total % 60).padStart(2, '0');
            return `${minutes}:${seconds}`;
        },

        // exo-style address choice: prefer a routable IPv4 on a direct or
        // known interface; a bare link-local fe80:: (no scope zone) can never
        // be dialed, so it ranks below everything.
        bestDeviceAddr(device) {
            const addrs = Array.isArray(device?.addrs) ? device.addrs : [];
            const preferred = ['manual', 'tb', 'thunderbolt', 'ethernet', 'tailscale'];
            const scored = addrs
                .filter((addr) => addr && addr.ip)
                .map((addr) => {
                    const ip = String(addr.ip);
                    let score = 0;
                    const rank = preferred.indexOf(String(addr.if_type || ''));
                    if (rank >= 0) score += 100 - rank;
                    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(ip)) score += 50;
                    if (ip.toLowerCase().startsWith('fe80:')) score -= 1000;
                    return { addr, score };
                })
                .sort((a, b) => b.score - a.score);
            return scored.length ? scored[0].addr : null;
        },

        coordinatorAddrFor(device) {
            const addr = this.bestDeviceAddr(device);
            if (!addr) return null;
            return `${addr.ip}:${device.http_port || 8000}`;
        },

        async beginJoinAsJoiner(device) {
            const target = this.coordinatorAddrFor(device);
            if (!target) {
                this.notify(
                    'error',
                    `No usable address for ${this.deviceName(device)} yet — try Add by IP.`,
                );
                return;
            }
            await this.beginJoinAddr(target, this.deviceName(device));
        },

        async beginJoinAddr(coordinatorAddr, targetName) {
            if (this.join.busy) return;
            this.join.busy = true;
            try {
                const snapshot = await this.apiFetch(CLUSTER_V2_API.pairJoin, {
                    method: 'POST',
                    body: JSON.stringify({ coordinator_addr: coordinatorAddr }),
                });
                this.join = {
                    ...this.join,
                    ...snapshot,
                    busy: false,
                    target_name: targetName || coordinatorAddr,
                };
                this.joinApprovedNotified = false;
                this.joinDeniedNotified = false;
                this.cancelPairing();
            } catch (error) {
                this.join.busy = false;
                this.notify(
                    'error',
                    error?.message || 'Could not reach that Mac',
                );
            }
        },

        async restartJoin() {
            // "Code expired — start again": same coordinator, fresh code.
            const addr = this.join.coordinator_addr;
            if (!addr) return;
            await this.beginJoinAddr(addr, this.join.target_name);
        },

        async cancelJoin(options = {}) {
            try {
                const snapshot = await this.apiFetch(
                    CLUSTER_V2_API.pairJoinCancel,
                    { method: 'POST' },
                );
                this.join = {
                    ...this.join,
                    ...(snapshot || { state: 'idle' }),
                    busy: false,
                    target_name: '',
                };
                if (!options.silent) {
                    this.notify('info', 'Join cancelled.');
                }
            } catch (error) {
                if (!options.silent) {
                    this.notify(
                        'error',
                        error?.message || 'Could not cancel the join',
                    );
                }
            }
        },

        // =====================================================================
        // Add by IP — the deterministic path when multicast can't reach the
        // other Mac (Thunderbolt pairs, filtered routers, Local Network off).
        // =====================================================================
        async submitManualPeer(options = {}) {
            if (this.manualBusy) return;
            const raw = (this.manualAddr || '').trim();
            const match = raw.match(/^(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?$/);
            if (!match) {
                this.manualError = 'Enter an IPv4 address like 192.168.1.50 or 192.168.1.50:8000.';
                return;
            }
            const ip = match[1];
            const port = match[2] ? parseInt(match[2], 10) : 8000;
            const octetsOk = ip.split('.').every((part) => Number(part) <= 255);
            if (!octetsOk || !(port >= 1 && port <= 65535)) {
                this.manualError = 'That address or port is out of range.';
                return;
            }
            this.manualBusy = true;
            this.manualError = '';
            try {
                const result = await this.apiFetch(CLUSTER_V2_API.manualDevice, {
                    method: 'POST',
                    body: JSON.stringify({ ip, port }),
                });
                await this.refreshDevices();
                if (result && result.verified) {
                    const name = result.peer?.friendly_name || ip;
                    this.notify('success', `Found ${name} at ${ip}.`);
                    if (options.beginJoin !== false) {
                        // First-cluster setup may join in either direction.
                        // An existing coordinator only discovers here: the
                        // new Mac initiates and this cluster approves its code.
                        await this.beginJoinAddr(`${ip}:${port}`, name);
                    }
                } else {
                    this.notify(
                        'warning',
                        'No oMLX node answered at that address yet — it stays on the list while we keep trying.',
                    );
                }
                this.manualAddr = '';
            } catch (error) {
                this.manualError =
                    error?.message || 'Could not add that address';
            } finally {
                this.manualBusy = false;
            }
        },

        // =====================================================================
        // Automatic checks — SSH, model presence, version parity, rdma_ctl,
        // benchmark, multicast self-test. Each is a row: spinner / pass / fail
        // with a fix.
        // =====================================================================
        startChecks() {
            this.stage = 'checks';
            this.runChecks();
        },

        async runChecks() {
            if (this.checks.running) return;
            this.stage = 'checks';
            this.checks.started = true;
            this.checks.running = true;
            const peers = this.pairedDevices();
            await Promise.all([
                ...peers.map((peer) => this.probePeer(peer)),
                this.refreshDiscoveryHealth(),
            ]);
            this.checks.running = false;
            this.checks.ranAt = Date.now();
        },

        sshTargetFor(device) {
            // Pairing enrollment records the SSH target; the devices payload
            // surfaces it as ssh_target on paired rows. Fall back to the
            // first verified probe address when no enrollment exists yet.
            if (device?.ssh_target) return String(device.ssh_target);
            const addrs = Array.isArray(device?.addrs) ? device.addrs : [];
            // A bare fe80:: link-local address has no scope id here, so SSH
            // to it has no route — prefer any routable address first.
            const usable = addrs.filter(
                (addr) => addr && addr.ip && !String(addr.ip).startsWith('fe80::'),
            );
            const first = usable[0] || addrs.find((addr) => addr && addr.ip);
            return first ? String(first.ip) : this.deviceName(device);
        },

        async probePeer(peer) {
            const ssh = this.sshTargetFor(peer);
            try {
                const result = await this.apiFetch(CLUSTER_V2_API.peerProbe, {
                    method: 'POST',
                    body: JSON.stringify({ ssh }),
                });
                this.checks.probes = {
                    ...this.checks.probes,
                    [peer.node_id]: { ok: true, ssh, result },
                };
            } catch (error) {
                this.checks.probes = {
                    ...this.checks.probes,
                    [peer.node_id]: {
                        ok: false,
                        ssh,
                        error: error?.message || 'Probe failed',
                    },
                };
            }
        },

        async runBenchmark() {
            if (this.checks.benchmarkRunning) return;
            if (!this.selectedModelPath) {
                this.checks.benchmark = {
                    ok: false,
                    error:
                        'Choose a downloaded model first. Calibration measures that real model on this exact cluster.',
                };
                return;
            }
            this.checks.benchmarkRunning = true;
            try {
                // "Run again" means a fresh probe: do not feed the previous
                // activation's profiles back into autoconfigure.
                this.planProposal = null;
                const proposal = await this.runPlan();
                if (!proposal) {
                    this.checks.benchmark = {
                        ok: false,
                        error: this.planError || 'Calibration failed',
                    };
                }
            } catch (error) {
                this.checks.benchmark = {
                    ok: false,
                    error: error?.message || 'Benchmark failed',
                };
            } finally {
                this.checks.benchmarkRunning = false;
            }
        },

        checkRows() {
            const peers = this.pairedDevices();
            const rows = [];
            const allPass = (list) => list.every(Boolean);

            // 1. SSH reachability (peer-probe per paired device).
            {
                const probed = peers.map(
                    (peer) => this.checks.probes[peer.node_id],
                );
                const running = this.checks.running && probed.some((p) => !p);
                const failures = peers.filter(
                    (peer) => this.checks.probes[peer.node_id]?.ok === false,
                );
                rows.push({
                    key: 'ssh',
                    label: 'SSH connection',
                    status: !this.checks.started
                        ? 'pending'
                        : running
                        ? 'running'
                        : peers.length && allPass(probed.map((p) => p?.ok))
                        ? 'pass'
                        : failures.length
                        ? 'fail'
                        : 'running',
                    detail: failures.length
                        ? `Can't reach ${failures
                              .map((peer) => this.deviceName(peer))
                              .join(', ')} over SSH.`
                        : 'Each Mac accepts the cluster key.',
                    fix: `On the failing Mac: System Settings → General → Sharing → turn on Remote Login, then press Re-run checks.`,
                });
            }

            // 2. Model presence (resolved once a model is chosen in the plan
            //    step; the wizard annotates presence per device there).
            {
                const model = this.selectedModel();
                let status = 'skipped';
                let detail = 'Checked automatically when you pick a model.';
                if (model) {
                    const holders = new Set(
                        (model.locations || []).map((loc) => loc.node_id),
                    );
                    const missing = this.allDevices().filter(
                        (device) =>
                            device.paired !== false &&
                            !holders.has(device.node_id) &&
                            !device.is_self &&
                            !(this.selfDevice() && holders.has('127.0.0.1')),
                    );
                    if (missing.length) {
                        status = 'fail';
                        detail = `${this.shortModelName(
                            model,
                        )} is missing on ${missing
                            .map((device) => this.deviceName(device))
                            .join(', ')}.`;
                    } else {
                        status = 'pass';
                        detail = `${this.shortModelName(model)} is on every Mac.`;
                    }
                }
                rows.push({
                    key: 'model',
                    label: 'Model on every Mac',
                    status,
                    detail,
                    fix: 'Open the model on the missing Mac and download it there, or let activation stage the files for you.',
                });
            }

            // 3. Version parity (from the discovery snapshot, live).
            {
                const mismatches = this.versionMismatches();
                rows.push({
                    key: 'version',
                    label: 'Matching oMLX versions',
                    status: mismatches.length ? 'fail' : 'pass',
                    detail: mismatches.length
                        ? mismatches
                              .map(
                                  (m) =>
                                      `${m.name} runs v${m.peerVersion}, this Mac runs v${m.selfVersion}.`,
                              )
                              .join(' ')
                        : 'Every Mac runs the same build.',
                    fix: 'Update the older Mac to the same oMLX build. App: it auto-updates. Brew: brew upgrade omlx. Source: pull the same commit on both Macs.',
                });
            }

            // 4. rdma_ctl / Thunderbolt fabric.
            {
                const fabricMembers = peers.filter(
                    (peer) => peer?.caps?.thunderbolt || peer?.caps?.jaccl,
                );
                if (!fabricMembers.length && !this.selfDevice()?.caps?.jaccl) {
                    rows.push({
                        key: 'rdma',
                        label: 'Thunderbolt RDMA (rdma_ctl)',
                        status: 'skipped',
                        detail:
                            'No Thunderbolt fabric detected — the TCP ring transport will be used instead.',
                        fix: '',
                    });
                } else {
                    const failing = fabricMembers.filter((peer) => {
                        const probe = this.checks.probes[peer.node_id];
                        if (!probe) return false;
                        if (!probe.ok) return true;
                        const rdma =
                            probe.result?.status?.transport?.rdma || {};
                        return !(rdma.devices || []).length;
                    });
                    rows.push({
                        key: 'rdma',
                        label: 'Thunderbolt RDMA (rdma_ctl)',
                        status: this.checks.running
                            ? 'running'
                            : failing.length
                            ? 'fail'
                            : 'pass',
                        detail: failing.length
                            ? `RDMA is not enabled on ${failing
                                  .map((peer) => this.deviceName(peer))
                                  .join(', ')}.`
                            : 'rdma_ctl reports devices on every Thunderbolt Mac.',
                        fix: 'Connect the Thunderbolt cable and run `rdma_ctl status` on every Mac. If you are not on macOS 27 and it reports disabled: shut down, hold the power button to enter Recovery, open Utilities → Terminal, run `rdma_ctl enable`, then restart and Re-run checks. Without RDMA the cluster falls back to the slower TCP ring.',
                    });
                }
            }

            // 5. Benchmark (explicit — it loads the chips for a few seconds).
            {
                const bench = this.checks.benchmark;
                rows.push({
                    key: 'benchmark',
                    label: 'Speed benchmark',
                    status: this.checks.benchmarkRunning
                        ? 'running'
                        : bench
                        ? bench.ok
                            ? 'pass'
                            : 'fail'
                        : !this.selectedModelPath
                        ? 'skipped'
                        : 'pending',
                    detail: bench
                        ? bench.ok
                            ? 'Measured compute and link speeds shape the layer split.'
                            : bench.error
                        : !this.selectedModelPath
                        ? 'Runs automatically after you choose a model.'
                        : 'Measuring the selected model so the split matches real speeds.',
                    fix: 'Calibration requires a selected model and awake peers. Choose the model, wake every Mac, then run it again.',
                });
            }

            // 6. Multicast / Local Network self-test (implemented at
            //    discovery_routes.py). Beacon loss is a warning, never a red
            //    failure: discovery degrades to Add by IP while pairing and
            //    the model split keep working — the footer only gates on
            //    SSH + versions, so a red row would contradict "All clear".
            {
                const health = this.discoveryHealth;
                let status = 'pending';
                let detail = 'Checks whether discovery beacons are arriving.';
                let fix =
                    'macOS is blocking local-network beacons. System Settings → Privacy & Security → Local Network → allow oMLX, then restart oMLX. Pairing still works via Add by IP while this is amber.';
                if (health) {
                    if (health.multicast_rx_within_5s) {
                        status = 'pass';
                        detail = 'Discovery beacons are flowing on this network.';
                    } else {
                        status = 'warn';
                        detail =
                            'No discovery beacons received in the last 5 seconds.';
                    }
                } else if (this.discoveryHealthUnsupported) {
                    status = 'skipped';
                    detail =
                        'This build does not report discovery health. Discovery itself may still work.';
                    fix = '';
                } else if (this.checks.running) {
                    status = 'running';
                }
                rows.push({
                    key: 'multicast',
                    label: t('cluster.v2.checks.beacon_label'),
                    status,
                    detail,
                    fix,
                });
            }

            return rows;
        },

        checksBlockingPass() {
            const rows = this.checkRows();
            const byKey = Object.fromEntries(rows.map((row) => [row.key, row]));
            return (
                byKey.ssh?.status === 'pass' &&
                byKey.version?.status === 'pass'
            );
        },

        // =====================================================================
        // Plan — visual layer split from the existing planner
        // =====================================================================
        enterPlan() {
            this.stage = 'plan';
            this.loadNodeRoles();
            if (!this.modelOptions.length && !this.modelsLoading) {
                this.loadModels();
            } else {
                // Models already cached — the recommendation badge still
                // needs its one catalogue call for this plan session.
                this.loadCatalogue();
            }
        },

        // =====================================================================
        // Per-node roles — how much of each Mac the cluster may take
        // =====================================================================
        async loadNodeRoles() {
            if (this.roleOptions.length) return;
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.nodeRoles);
                this.roleOptions = payload?.roles || [];
            } catch (error) {
                // The mirrored fallback constants keep the budget labels
                // rendering; planning itself re-derives roles server-side.
            }
        },

        roleSpecFor(key) {
            return (
                this.roleOptions.find((role) => role.key === key) ||
                CLUSTER_V2_ROLE_FALLBACK[key] ||
                CLUSTER_V2_ROLE_FALLBACK.headless
            );
        },

        roleLabel(key) {
            return this.roleSpecFor(key).label || key;
        },

        nodeRole(nodeId, isSelf) {
            // The default is unchanged: the Mac whose display is in front of
            // the user keeps a workstation reserve; everything else is
            // headless (also the server default).
            return this.nodeRoles[nodeId] || (isSelf ? 'workstation' : 'headless');
        },

        setNodeRole(device, role) {
            if (!device || this.planLoading) return;
            if (this.nodeRole(device.node_id, !!device.is_self) === role) return;
            this.nodeRoles = { ...this.nodeRoles, [device.node_id]: role };
            this.planFitFailure = null;
            if (this.selectedModelPath) this.runPlan();
        },

        // The paired Macs that will receive layers (this Mac included).
        planRoleDevices() {
            return this.allDevices().filter(
                (device) => device.paired && this.deviceRamGb(device),
            );
        },

        reserveBytesFor(roleKey, capacityBytes) {
            // Mirrors NodeRole.reserve_for in omlx/cluster/node_role.py using
            // the server-provided reserve_bytes/reserve_fraction pair (or the
            // synced fallback mirror above): max(absolute floor, fractional
            // headroom), always leaving the model at least 1 GiB.
            const spec = this.roleSpecFor(roleKey);
            const gib = 1024 ** 3;
            const reserve = Math.max(
                Number(spec.reserve_bytes || 0),
                Math.floor(capacityBytes * Number(spec.reserve_fraction || 0)),
            );
            return Math.min(reserve, Math.max(0, capacityBytes - gib));
        },

        usableGbLabel(device) {
            const capacity = (this.deviceRamGb(device) || 0) * 1024 ** 3;
            if (!capacity) return '';
            const role = this.nodeRole(device.node_id, !!device.is_self);
            const usable = Math.max(
                0,
                capacity - this.reserveBytesFor(role, capacity),
            );
            return `${Math.round(usable / 1024 ** 3)} GB usable as ${this.roleLabel(role)}`;
        },

        // What flipping every workstation node to headless would free up.
        headlessGainBytes() {
            return this.planRoleDevices().reduce((gain, device) => {
                const capacity = (this.deviceRamGb(device) || 0) * 1024 ** 3;
                const role = this.nodeRole(device.node_id, !!device.is_self);
                if (role !== 'workstation') return gain;
                return (
                    gain +
                    this.reserveBytesFor('workstation', capacity) -
                    this.reserveBytesFor('headless', capacity)
                );
            }, 0);
        },

        // /plan 400s with "model does not fit the supplied per-node budgets
        // (at least N additional bytes required)" — turn the number into
        // guidance instead of a dead end.
        parseFitFailure(message) {
            if (!/per-node budgets/.test(message || '')) return null;
            const match = /at least (\d+) additional bytes/.exec(message || '');
            if (!match) return null;
            const shortfallBytes = Number(match[1]);
            return {
                shortfallBytes,
                canFixWithHeadless:
                    shortfallBytes > 0 &&
                    this.headlessGainBytes() >= shortfallBytes,
            };
        },

        fitShortfallLabel() {
            const bytes = this.planFitFailure?.shortfallBytes;
            return bytes ? `${(bytes / 1024 ** 3).toFixed(1)} GiB` : '';
        },

        // One explicit click: roles never flip on their own.
        async switchAllToHeadless() {
            const flipped = { ...this.nodeRoles };
            for (const device of this.planRoleDevices()) {
                flipped[device.node_id] = 'headless';
            }
            this.nodeRoles = flipped;
            this.planFitFailure = null;
            await this.runPlan();
        },

        inventoryHosts() {
            const hosts = [
                {
                    node_id: this.selfDevice()?.node_id || 'coordinator',
                    ssh: '127.0.0.1',
                },
            ];
            for (const peer of this.pairedDevices()) {
                hosts.push({
                    node_id: peer.node_id,
                    ssh: this.sshTargetFor(peer),
                    python_executable:
                        this.checks.probes[peer.node_id]?.result?.status
                            ?.runtime?.python_executable || undefined,
                });
            }
            return hosts;
        },

        async loadModels() {
            this.modelsLoading = true;
            this.modelsError = '';
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.models, {
                    method: 'POST',
                    body: JSON.stringify({ hosts: this.inventoryHosts() }),
                });
                this.modelOptions = payload?.models || [];
            } catch (error) {
                this.modelsError =
                    error?.message || 'Could not list downloaded models';
            } finally {
                this.modelsLoading = false;
            }
            // Catalogue advice (recommended strategy pill) rides on the model
            // list — fetched once per plan session, silently.
            this.loadCatalogue();
        },

        filteredModels() {
            const query = (this.modelSearch || '').trim().toLowerCase();
            if (!query) return this.modelOptions;
            return this.modelOptions.filter((model) =>
                `${model.id || ''} ${model.model_path || ''} ${
                    model.display_name || ''
                }`
                    .toLowerCase()
                    .includes(query),
            );
        },

        selectedModel() {
            return (
                this.modelOptions.find(
                    (model) => model.model_path === this.selectedModelPath,
                ) || null
            );
        },

        // A name a person recognizes: the inventory's display_name when the
        // server provides one; otherwise a model_path cleaned of Hugging Face
        // cache internals — snapshot-hash tail segments (40–64 hex chars) and
        // the models--org--name directory encoding.
        displayModelName(model) {
            const display = String(model?.display_name || '').trim();
            if (display) return display;
            const raw = String(
                model?.model_path || model?.model || model?.id || '',
            );
            const segments = raw.split('/').filter(Boolean);
            while (
                segments.length &&
                /^[0-9a-f]{40,64}$/i.test(segments[segments.length - 1])
            ) {
                segments.pop();
            }
            if (segments[segments.length - 1] === 'snapshots') segments.pop();
            let name = segments.pop() || raw || 'this model';
            const hub = /^models--([^/]+?)--(.+)$/.exec(name);
            if (hub) name = `${hub[1]}/${hub[2]}`;
            return name;
        },

        shortModelName(model) {
            return this.displayModelName(model);
        },

        // Denominator is the Macs that will actually run the split (this Mac
        // + paired peers) — discovered-but-unpaired devices never receive
        // layers, so counting them misreported "on 1 of 3 Macs".
        modelPresenceLabel(model) {
            const holders = new Set(
                (model?.locations || []).map((loc) => loc.node_id),
            );
            const total =
                (this.selfDevice() ? 1 : 0) + this.pairedDevices().length;
            if (!total) return '';
            const have = Math.min(holders.size, total);
            if (have >= total) return t('cluster.v2.models.on_every_mac');
            return t('cluster.v2.models.partial')
                .replace('{have}', String(have))
                .replace('{total}', String(total));
        },

        async selectModel(model) {
            if (!model || this.planLoading) return;
            this.selectedModelPath = model.model_path;
            const modelLimit = Math.max(
                1,
                Number(model.model_context_length || 1_048_576),
            );
            this.targetContextTokens = Math.min(
                Math.max(1, Number(this.targetContextTokens) || 32768),
                modelLimit,
            );
            this.plan = null;
            this.planProposal = null;
            this.checks.benchmark = null;
            this.normalizePlanStrategy();
            await this.runPlan();
        },

        // =====================================================================
        // Execution strategy — auto / tensor / pipeline, server-recommended
        // =====================================================================
        executionProfileOptions() {
            return [
                {
                    key: 'interactive',
                    label: 'Interactive',
                    limits: '4 decode · 2 prompt · batch 2',
                    detail: 'Lower queueing and memory use',
                },
                {
                    key: 'balanced',
                    label: 'Balanced',
                    limits: '8 decode · 4 prompt · batch 4',
                    detail: 'Default mix of latency and throughput',
                },
                {
                    key: 'throughput',
                    label: 'Throughput',
                    limits: '16 decode · 8 prompt · batch 8',
                    detail: 'Wider automatic batches when requests overlap',
                },
            ];
        },

        contextReservationOptions() {
            const selectedLimit = Math.max(
                1,
                Number(this.selectedModel()?.model_context_length || 1_048_576),
            );
            const standard = [8192, 32768, 131072, 262144, 524288, 1_048_576];
            const values = standard.filter((value) => value <= selectedLimit);
            if (!values.includes(selectedLimit)) values.push(selectedLimit);
            return [...new Set(values)].sort((left, right) => left - right);
        },

        contextReservationLabel(tokens) {
            const value = Math.max(1, Number(tokens) || 0);
            if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2).replace(/\.00$/, '')}M tokens`;
            return `${Math.round(value / 1024)}K tokens`;
        },

        setTargetContextTokens(value) {
            const parsed = Math.max(1, Number(value) || 32768);
            if (parsed === this.targetContextTokens || this.planLoading) return;
            this.targetContextTokens = parsed;
            if (this.selectedModelPath) this.runPlan();
        },

        setExecutionProfile(key) {
            if (
                this.planLoading ||
                !this.executionProfileOptions().some(
                    (option) => option.key === key,
                ) ||
                this.executionProfile === key
            ) {
                return;
            }
            this.executionProfile = key;
            // Catalogue recommendations also use the workload profile, so do
            // not retain a recommendation computed for the old limits.
            this.catalogueModels = null;
            this.catalogueFailed = false;
            if (this.modelOptions.length) this.loadCatalogue();
            if (this.selectedModelPath) this.runPlan();
        },

        catalogueEntryForModel() {
            if (!Array.isArray(this.catalogueModels)) return null;
            const model = this.selectedModel();
            if (!model) return null;
            return (
                this.catalogueModels.find(
                    (entry) => entry.model_path === model.model_path,
                ) || null
            );
        },

        // One 'Recommended' pill, on exactly one option: the strategy the
        // catalogue names for the picked model, or — when the catalogue could
        // not advise — tensor iff every member sits on a fast (jaccl /
        // Thunderbolt) transport, mirroring resolvedBackend().
        recommendedStrategy() {
            const entry = this.catalogueEntryForModel();
            if (entry) {
                if (entry.strategy === 'tensor') return 'tensor';
                if (entry.strategy === 'pipeline') return 'pipeline';
                // 'hybrid' / 'single node' cannot be expressed by the picker.
                return 'auto';
            }
            if (
                this.catalogueFailed ||
                Array.isArray(this.catalogueModels)
            ) {
                // Fallback heuristic: tensor only pays off when every member
                // sits on a fast (jaccl / Thunderbolt) transport — the same
                // rule resolvedBackend() applies; anything else, auto.
                return this.resolvedBackend() === 'jaccl' ? 'tensor' : 'auto';
            }
            return '';
        },

        strategyOptions() {
            const entry = this.catalogueEntryForModel();
            const nodeCount = this.planNodes().length;
            const tensorUnsupported =
                !!entry && entry.supports_tensor_parallel === false;
            const pipelineUnsupported =
                !!entry && entry.supports_pipeline === false;
            const tensorDisabled = nodeCount < 2 || tensorUnsupported;
            const phaseFit = this.phaseReplicaFitEstimate();
            const phaseDisabled = nodeCount !== 2 || phaseFit?.fits === false;
            return [
                {
                    key: 'auto',
                    label: t('cluster.v2.strategy.auto'),
                    disabled: false,
                    disabledReason: '',
                },
                {
                    key: 'tensor',
                    label: t('cluster.v2.strategy.tensor'),
                    disabled: tensorDisabled,
                    disabledReason: !tensorDisabled
                        ? ''
                        : nodeCount < 2
                        ? t('cluster.v2.strategy.tensor_needs_two')
                        : t('cluster.v2.strategy.tensor_unsupported'),
                },
                {
                    key: 'pipeline',
                    label: t('cluster.v2.strategy.pipeline'),
                    disabled: pipelineUnsupported,
                    disabledReason: pipelineUnsupported
                        ? t('cluster.v2.strategy.pipeline_unsupported')
                        : '',
                },
                {
                    key: 'disaggregated',
                    label: t('cluster.v2.strategy.disaggregated'),
                    disabled: phaseDisabled,
                    disabledReason: phaseDisabled
                        ? nodeCount !== 2
                            ? 'Phase split currently requires exactly two Macs.'
                            : 'The complete model does not fit on both Macs.'
                        : '',
                },
            ];
        },

        selectedStrategyOption() {
            return (
                this.strategyOptions().find(
                    (option) => option.key === this.planStrategy,
                ) || { key: 'auto', disabledReason: '' }
            );
        },

        strategyHint() {
            return t(`cluster.v2.strategy.hint.${this.planStrategy}`);
        },

        setPlanStrategy(key) {
            const option = this.strategyOptions().find(
                (item) => item.key === key,
            );
            if (!option || option.disabled || this.planLoading) return;
            if (this.planStrategy === key) return;
            this.planStrategy = key;
            if (key === 'disaggregated') {
                this.phasePrefillRank = Math.min(
                    1,
                    Math.max(0, Number(this.phasePrefillRank) || 0),
                );
            }
            if (this.selectedModelPath) this.runPlan();
        },

        phaseRoleDevices() {
            return this.planRoleDevices().filter((_device, index) => index < 2);
        },

        phaseRecommendedPrefillRank() {
            const profiles = Array.isArray(this.plan?.performance_profiles)
                ? this.plan.performance_profiles
                : [];
            const measured = profiles
                .map((profile) => ({
                    rank: Number(profile?.rank),
                    rate: Number(profile?.prefill_weight_bytes_per_second || 0),
                }))
                .filter(
                    (item) =>
                        [0, 1].includes(item.rank) &&
                        Number.isFinite(item.rate) &&
                        item.rate > 0,
                )
                .sort((left, right) => right.rate - left.rate);
            if (measured.length) return measured[0].rank;
            const devices = this.phaseRoleDevices();
            const headless = devices.findIndex(
                (device) =>
                    this.nodeRole(device?.node_id, !!device?.is_self) === 'headless',
            );
            return headless >= 0 ? headless : Math.min(1, devices.length - 1);
        },

        phaseRecommendationReason() {
            const profiles = Array.isArray(this.plan?.performance_profiles)
                ? this.plan.performance_profiles
                : [];
            return profiles.length
                ? 'Recommended from the measured prefill profile.'
                : 'Recommended to keep the coordinator focused on decode and API work.';
        },

        phaseDeviceMeta(device) {
            return [
                this.deviceChipLabel(device),
                this.deviceRamLabel(device),
                this.usableGbLabel(device),
            ]
                .filter(Boolean)
                .join(' · ');
        },

        phaseReplicaFitEstimate() {
            const entry = this.catalogueEntryForModel();
            const weightBytes = Number(entry?.weight_bytes || 0);
            const devices = this.phaseRoleDevices();
            if (!(weightBytes > 0) || devices.length !== 2) return null;
            const usable = devices.map((device) => {
                const capacity = (this.deviceRamGb(device) || 0) * (1024 ** 3);
                const role = this.nodeRole(device.node_id, !!device.is_self);
                return Math.max(
                    0,
                    capacity - this.reserveBytesFor(role, capacity),
                );
            });
            return {
                fits: usable.every((value) => value >= weightBytes),
                weightBytes,
                smallestUsableBytes: Math.min(...usable),
            };
        },

        phaseFitSummary() {
            if (this.planIsDisaggregated()) {
                const context = Number(this.plan?.target_context_tokens || 0);
                return `Signed fit verified on both Macs${
                    context ? ` at ${context.toLocaleString()} tokens` : ''
                }.`;
            }
            const estimate = this.phaseReplicaFitEstimate();
            if (estimate && !estimate.fits) {
                return `Unavailable: complete weights need ${(
                    estimate.weightBytes /
                    1024 ** 3
                ).toFixed(1)} GiB, above the smallest Mac's usable ${(
                    estimate.smallestUsableBytes /
                    1024 ** 3
                ).toFixed(1)} GiB.`;
            }
            return 'Final approval verifies complete weights plus the selected context reservation on both Macs.';
        },

        setPhasePrefillRank(rank) {
            const value = Number(rank);
            if (![0, 1].includes(value) || this.planLoading) return;
            if (this.phasePrefillRank === value) return;
            this.phasePrefillRank = value;
            if (this.selectedModelPath && this.planStrategy === 'disaggregated') {
                this.runPlan();
            }
        },

        phaseRoleForRank(rank) {
            return Number(rank) === Number(this.phasePrefillRank)
                ? 'Prefill'
                : 'Decode';
        },

        // A model change (or catalogue arrival) can invalidate the current
        // pick — e.g. pipeline selected for a pipeline-incapable model.
        normalizePlanStrategy() {
            const option = this.strategyOptions().find(
                (item) => item.key === this.planStrategy,
            );
            if (option?.disabled) this.planStrategy = 'auto';
        },

        // Advisory only: on failure the badge falls back to the transport
        // heuristic. No toasts — a missing recommendation is not an error.
        async loadCatalogue() {
            if (
                this.catalogueLoading ||
                this.catalogueModels !== null ||
                this.catalogueFailed
            ) {
                return;
            }
            const candidates = this.modelOptions
                .filter((model) => model?.model_path)
                .map((model) => {
                    const sourceLocation =
                        (model.locations || []).find(
                            (loc) => loc.ssh === model.model_source,
                        ) || {};
                    return {
                        id: String(model.id || model.model_path),
                        model_path: model.model_path,
                        model_source: model.model_source || '127.0.0.1',
                        model_source_python:
                            sourceLocation.python_executable || undefined,
                        source_node_id: model.source_node_id || '',
                        model_context_length:
                            model.model_context_length || undefined,
                    };
                });
            if (!candidates.length) return;
            this.catalogueLoading = true;
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.catalogue, {
                    method: 'POST',
                    body: JSON.stringify({
                        nodes: this.planNodes(),
                        models: candidates,
                        execution_profile: this.executionProfile,
                    }),
                });
                this.catalogueModels = payload?.models || [];
                this.normalizePlanStrategy();
            } catch (error) {
                this.catalogueFailed = true;
            } finally {
                this.catalogueLoading = false;
            }
        },

        // Pure tensor plans give every node all layers. Hybrid plans give each
        // TP group one contiguous stage, then split every layer in that stage.
        planIsTensor() {
            return (this.plan?.tensor_parallel_size || 1) > 1;
        },

        planIsDisaggregated() {
            return this.plan?.serving_mode === 'disaggregated';
        },

        phasePlanRole(assignment) {
            if (!this.planIsDisaggregated()) return '';
            return Number(assignment?.rank) === Number(this.plan?.prefill_rank)
                ? 'Prefill'
                : 'Decode';
        },

        planPipelineStages() {
            return Math.max(1, Number(this.plan?.pipeline_stages || 1));
        },

        planIsHybrid() {
            return this.planIsTensor() && this.planPipelineStages() > 1;
        },

        tensorGroupRows(assignment = null) {
            if (!assignment) return this.planAssignments();
            return this.planAssignments().filter(
                (row) =>
                    Number(row.start_layer) === Number(assignment.start_layer) &&
                    Number(row.end_layer) === Number(assignment.end_layer),
            );
        },

        tensorShareLabel(assignment = null) {
            const rows = this.tensorGroupRows(assignment);
            const weights = rows.map((row) =>
                Number(row.tensor_parallel_shard_weight || 1),
            );
            const total = weights.reduce((sum, value) => sum + value, 0);
            const weight = Number(
                assignment?.tensor_parallel_shard_weight || 0,
            );
            if (assignment && total > 0 && new Set(weights).size > 1) {
                return `${weight}/${total} tensor rows · layers [${assignment.start_layer}, ${assignment.end_layer})`;
            }
            const share = t('cluster.v2.split.tensor_share').replace(
                '{count}',
                String(this.plan?.tensor_parallel_size || 1),
            );
            return this.planIsHybrid() && assignment
                ? `${share} · layers [${assignment.start_layer}, ${assignment.end_layer})`
                : share;
        },

        tensorSharePercent(assignment) {
            const rows = this.tensorGroupRows(assignment);
            const total = rows.reduce(
                (sum, row) =>
                    sum + Number(row.tensor_parallel_shard_weight || 1),
                0,
            );
            if (total <= 0) return 0;
            const tensorFraction =
                Number(assignment?.tensor_parallel_shard_weight || 1) / total;
            if (!this.planIsHybrid()) return 100 * tensorFraction;
            const stageFraction =
                Number(assignment?.layer_count || 0) /
                Math.max(this.planTotalLayers(), 1);
            return 100 * stageFraction * tensorFraction;
        },

        tensorQualification() {
            const value = this.plan?.tensor_parallel_qualification;
            return value && typeof value === 'object' ? value : null;
        },

        tensorQualificationIsPersistent() {
            return this.tensorQualification()?.source === 'persistent';
        },

        tensorRecommendation() {
            const value = this.planProposal?.tp_layout_recommendation;
            return value && typeof value === 'object' ? value : null;
        },

        tensorRecommendationNeedsCalibration() {
            return this.tensorRecommendation()?.state === 'calibration_required';
        },

        tensorRecommendationIsRejected() {
            return this.tensorRecommendation()?.state === 'rejected';
        },

        tensorQualificationLabel() {
            if (this.planIsHybrid()) {
                return `Hybrid TP×pipeline · equal safe tensor split in each of ${this.planPipelineStages()} stages`;
            }
            const qualification = this.tensorQualification();
            const weights = this.planAssignments().map((assignment) =>
                Number(assignment.tensor_parallel_shard_weight || 1),
            );
            const vector = weights.join(':');
            if (qualification?.source === 'persistent') {
                return `Performance-qualified ${vector} split`;
            }
            if (qualification?.source === 'environment_override') {
                return `Experimental ${vector} override`;
            }
            const recommendation = this.tensorRecommendation();
            if (recommendation?.state === 'calibration_required') {
                const candidate = (
                    recommendation.recommended_weights || []
                ).join(':');
                return `Recommended ${candidate} · calibration required`;
            }
            if (recommendation?.state === 'rejected') {
                const candidate = (
                    recommendation.recommended_weights || []
                ).join(':');
                return `Rejected ${candidate} · parity or performance failed`;
            }
            return `Equal ${vector} safe fallback`;
        },

        tensorCaptionLabel() {
            if (this.planIsHybrid()) {
                return `${this.plan?.tensor_parallel_size || 1}-way tensor × ${this.planPipelineStages()} pipeline stages`;
            }
            return t('cluster.v2.split.tensor_caption').replace(
                '{count}',
                String(this.plan?.tensor_parallel_size || 1),
            );
        },

        planNodes() {
            const nodes = [];
            const self = this.selfDevice();
            if (self) {
                nodes.push({
                    node_id: self.node_id,
                    capacity_bytes:
                        (this.deviceRamGb(self) || 0) * 1024 ** 3,
                    // The Mac whose display is in front of the user keeps a
                    // workstation reserve unless the plan step says otherwise.
                    role: this.nodeRole(self.node_id, true),
                    memory_guard_tier: 'balanced',
                    accelerator: 'metal',
                });
            }
            for (const peer of this.pairedDevices()) {
                nodes.push({
                    node_id: peer.node_id,
                    capacity_bytes:
                        (this.deviceRamGb(peer) || 0) * 1024 ** 3,
                    role: this.nodeRole(peer.node_id, false),
                    memory_guard_tier: 'balanced',
                    accelerator: 'metal',
                });
            }
            return nodes.filter((node) => node.capacity_bytes > 0);
        },

        async runPlan() {
            if (!this.selectedModelPath || this.planLoading) return null;
            const revision = ++this.planRequestRevision;
            const previousActivation = this.planProposal?.activation;
            this.planLoading = true;
            this.planError = '';
            this.planFitFailure = null;
            this.plan = null;
            this.planProposal = null;
            try {
                const model = this.selectedModel();
                const priorPerformance = new Map(
                    (previousActivation?.nodes || [])
                        .filter((node) => node?.node_id && node?.performance)
                        .map((node) => [node.node_id, node.performance]),
                );
                const nodes = this.planNodes().map((node) => ({
                    ...node,
                    ...(priorPerformance.has(node.node_id)
                        ? { performance: priorPerformance.get(node.node_id) }
                        : {}),
                }));
                const body = {
                    model_path: this.selectedModelPath,
                    nodes,
                    hosts: this.deploymentHosts(),
                    execution_profile: this.executionProfile,
                    prompt_cache_ssd: this.promptCacheSsd,
                    prompt_cache_ssd_max_bytes:
                        Math.max(1, Number(this.promptCacheSsdMaxGiB) || 20) * (1024 ** 3),
                    prefer: 'speed',
                    strategy: this.planStrategy,
                    detect_transports: true,
                    preflight: true,
                    auto_tune: true,
                    measure_performance: true,
                    target_context_tokens: Math.max(
                        1,
                        Number(this.targetContextTokens) || 32768,
                    ),
                };
                if (this.planStrategy === 'disaggregated') {
                    body.prefill_rank = Number(this.phasePrefillRank);
                    body.decode_rank = 1 - Number(this.phasePrefillRank);
                    body.measure_performance = false;
                }
                if (
                    model?.model_source &&
                    model.model_source !== '127.0.0.1'
                ) {
                    body.model_source = model.model_source;
                }
                const sourceLocation = (model?.locations || []).find(
                    (location) => location?.ssh === model?.model_source,
                );
                const sourcePython =
                    sourceLocation?.python_executable ||
                    model?.python_executable;
                if (sourcePython) body.model_source_python = sourcePython;

                const proposal = await this.apiFetch(
                    CLUSTER_V2_API.autoconfigure,
                    {
                        method: 'POST',
                        body: JSON.stringify(body),
                    },
                );
                if (revision !== this.planRequestRevision) return null;
                if (
                    !proposal?.plan ||
                    !proposal?.activation ||
                    typeof proposal.activation !== 'object'
                ) {
                    throw new Error(
                        'Automatic setup did not return a signed activation proposal.',
                    );
                }
                if (
                    proposal.activation.approved_placement !==
                    proposal.plan.placement_signature
                ) {
                    throw new Error(
                        'Automatic setup returned a plan and activation with different signatures.',
                    );
                }
                this.planProposal = proposal;
                this.plan = proposal.plan;
                const probe = proposal.performance_probe;
                this.checks.benchmark =
                    probe?.ok === true ||
                    probe?.status === 'phase_probe_required'
                        ? { ok: true, result: probe }
                        : {
                              ok: false,
                              result: probe || null,
                              error:
                                  probe?.reason ||
                                  'The server did not complete performance calibration.',
                          };
                return proposal;
            } catch (error) {
                if (revision !== this.planRequestRevision) return null;
                this.planError =
                    error?.message || 'Could not build the layer split';
                this.planFitFailure = this.parseFitFailure(this.planError);
                this.checks.benchmark = {
                    ok: false,
                    error: this.planError,
                };
                return null;
            } finally {
                if (revision === this.planRequestRevision) {
                    this.planLoading = false;
                }
            }
        },

        planAssignments() {
            const assignments = this.plan?.assignments;
            return Array.isArray(assignments) ? assignments : [];
        },

        planTotalLayers() {
            if (this.planIsDisaggregated()) {
                return this.planAssignments().reduce(
                    (largest, assignment) =>
                        Math.max(largest, Number(assignment.layer_count || 0)),
                    0,
                );
            }
            return this.planAssignments().reduce(
                (sum, assignment) => sum + (assignment.layer_count || 0),
                0,
            );
        },

        planNodeName(assignment) {
            const device = this.allDevices().find(
                (item) => item.node_id === assignment.node_id,
            );
            return device
                ? this.deviceName(device)
                : assignment.node_id || `Rank ${assignment.rank}`;
        },

        resolvedBackend() {
            const serverBackend =
                this.planProposal?.activation?.backend ||
                this.planProposal?.backend;
            if (['jaccl', 'jaccl-ring', 'ring'].includes(serverBackend)) {
                return serverBackend;
            }
            const members = [this.selfDevice(), ...this.pairedDevices()].filter(
                Boolean,
            );
            const allJaccl =
                members.length > 1 &&
                members.every((device) => device?.caps?.jaccl);
            return allJaccl ? 'jaccl' : 'ring';
        },

        backendLabel() {
            return this.resolvedBackend().startsWith('jaccl')
                ? 'JACCL · Thunderbolt RDMA'
                : 'TCP ring';
        },

        deploymentFabricLabel() {
            if (this.configuredDeployment()?.serving_mode === 'disaggregated') {
                return this.resolvedBackend().startsWith('jaccl')
                    ? 'Inference: Prefill → Decode over Thunderbolt RDMA'
                    : 'Inference: Prefill → Decode over TCP ring';
            }
            const backend = String(
                this.configuredDeployment()?.backend || this.resolvedBackend(),
            );
            return backend.startsWith('jaccl')
                ? 'Inference: JACCL over Thunderbolt RDMA'
                : 'Inference: TCP ring';
        },

        deploymentHosts() {
            const hosts = [];
            const self = this.selfDevice();
            if (self) {
                hosts.push({
                    node_id: self.node_id,
                    ssh: '127.0.0.1',
                    ips: (self.addrs || [])
                        .map((addr) => addr?.ip)
                        .filter(Boolean),
                    rdma: [],
                });
            }
            for (const peer of this.pairedDevices()) {
                hosts.push({
                    node_id: peer.node_id,
                    ssh: this.sshTargetFor(peer),
                    ips: (peer.addrs || [])
                        .map((addr) => addr?.ip)
                        .filter(Boolean),
                    rdma: [],
                    python_executable:
                        this.checks.probes[peer.node_id]?.result?.status
                            ?.runtime?.python_executable || undefined,
                });
            }
            return hosts.filter((host) => host.ips.length || host.ssh);
        },

        // The server-produced activation is immutable client input from here
        // onward. Rebuilding any of its nodes, host order, backend, profiles or
        // TP degree in the browser would invalidate the signed preview.
        activationRequestBody() {
            const activation = this.planProposal?.activation;
            return activation && typeof activation === 'object'
                ? activation
                : null;
        },

        // =====================================================================
        // Auto-staging — "copied at activation", literally (phase 1 of 2)
        // =====================================================================
        // Plan members that do not hold a complete copy of the selected
        // model. Heuristic only — the server re-verifies file-by-file and
        // size-skips whatever is already there. A location counts as complete
        // when its estimated_size matches the largest location (the same
        // proxy merge_model_inventories uses to pick the source).
        nodesMissingModel() {
            const model = this.selectedModel();
            if (!model) return [];
            const locations = model.locations || [];
            // No inventory locations means we cannot tell who holds what —
            // fall back to the pre-staging direct activation.
            if (!locations.length) return [];
            const full = Math.max(
                0,
                ...locations.map((loc) => loc.estimated_size || 0),
            );
            const holders = new Set(
                locations
                    .filter((loc) => (loc.estimated_size || 0) >= full)
                    .map((loc) => loc.node_id),
            );
            const activationNodes = this.activationRequestBody()?.nodes;
            const members = Array.isArray(activationNodes)
                ? activationNodes
                : [];
            return members.filter(
                (node) => !holders.has(node.node_id),
            );
        },

        needsStaging() {
            return this.nodesMissingModel().length > 0;
        },

        async activatePlan() {
            const activation = this.activationRequestBody();
            if (!this.plan || !activation || this.activateBusy) return;
            this.activateBusy = true;
            if (this.needsStaging()) {
                // Phase 1 returns after the POST; the 1 Hz poller owns the
                // flow from here and chains into phase 2 on completion.
                await this.stageModelToPeers(activation);
                return;
            }
            // Every plan Mac already has the model — today's direct path.
            await this.postActivation(activation);
        },

        async stageModelToPeers(activation) {
            this.stagingError = '';
            this.stagingActivation = activation;
            let job;
            try {
                job = await this.apiFetch(CLUSTER_V2_API.stage, {
                    method: 'POST',
                    body: JSON.stringify({
                        activation,
                        parallel: 4,
                    }),
                });
            } catch (error) {
                this.stopStagingPoll();
                if (error?.status === 409) {
                    // Signature drift between plan and stage — identical
                    // handling to activation's 409 below.
                    this.notify(
                        'warning',
                        'The plan changed since you reviewed it — rebuilding it now.',
                    );
                    this.activateBusy = false;
                    this.stagingActivation = null;
                    await this.runPlan();
                } else if (error?.status === 404) {
                    // A server older than /stage: degrade to the pre-staging
                    // behavior instead of bricking activation — the
                    // activation preflight reports whatever is still missing.
                    this.notify('warning', t('cluster.v2.staging.unsupported'));
                    await this.postActivation(activation);
                } else {
                    this.stagingActivation = null;
                    this.failStaging(
                        error?.message || t('cluster.v2.staging.failed'),
                    );
                    this.activateBusy = false;
                }
                return;
            }
            this.stagingJob = job;
            this.stagingTimer = setInterval(
                () => this.pollStagingJob(),
                CLUSTER_V2_POLL_MS,
            );
        },

        async pollStagingJob() {
            const jobId = this.stagingJob?.job_id;
            if (!jobId) {
                this.stopStagingPoll();
                return;
            }
            let snapshot;
            try {
                snapshot = await this.apiFetch(
                    CLUSTER_V2_API.stageJob(jobId),
                );
            } catch (error) {
                // 404 = the coordinator restarted and the in-memory job is
                // gone. Re-POSTing /stage is safe (verified files skip), so
                // pressing Activate again is the resume.
                this.stopStagingPoll();
                this.failStaging(
                    error?.status === 404
                        ? t('cluster.v2.staging.lost')
                        : error?.message || t('cluster.v2.staging.failed'),
                );
                this.activateBusy = false;
                return;
            }
            this.stagingJob = snapshot;
            if (snapshot.status === 'completed') {
                this.stopStagingPoll();
                await this.postActivation(this.stagingActivation); // phase 2
            } else if (snapshot.status === 'failed') {
                // Other nodes may still have completed; per-node errors stay
                // visible in stagingJob.nodes next to the banner. Pressing
                // Activate retries — only missing files move again.
                this.stopStagingPoll();
                this.failStaging(
                    snapshot.error || t('cluster.v2.staging.failed'),
                );
                this.activateBusy = false;
            }
        },

        stopStagingPoll() {
            if (this.stagingTimer) {
                clearInterval(this.stagingTimer);
                this.stagingTimer = null;
            }
        },

        failStaging(message) {
            this.stagingError = message || t('cluster.v2.staging.failed');
            this.notify('error', this.stagingError);
        },

        // Client-side dismiss only — there is no cancel endpoint. The copy
        // finishes server-side regardless, harmlessly: the next attempt
        // size-verifies and skips whatever already landed.
        dismissStaging() {
            this.stopStagingPoll();
            this.stagingJob = null;
            this.stagingActivation = null;
            this.stagingError = '';
            this.activateBusy = false;
        },

        stagingNodes() {
            return Object.values(this.stagingJob?.nodes || {});
        },

        // Per-node rows with file counts are the honest display: later nodes
        // report bytes_total 0 until their turn starts and bytes jump per
        // completed file, so a single overall byte % would lie.
        stagingNodeLabel(node) {
            const status = node?.status || 'queued';
            if (status === 'ready') {
                // ready with nothing copied = the node already held the
                // complete model (the copying blip is sub-second).
                return (node.bytes_total || 0) > 0
                    ? t('cluster.v2.staging.ready')
                    : t('cluster.v2.staging.already_there');
            }
            if (status === 'failed') {
                return node.error || t('cluster.v2.staging.failed');
            }
            if (status === 'copying') {
                if (!(node.files_total > 0)) {
                    return t('cluster.v2.staging.checking');
                }
                return t('cluster.v2.staging.copying')
                    .replace('{done}', String(node.files_completed || 0))
                    .replace('{total}', String(node.files_total || 0));
            }
            return t('cluster.v2.staging.waiting');
        },

        stagingOverallLabel() {
            const nodes = this.stagingNodes();
            const done = nodes.filter(
                (node) => node.status === 'ready',
            ).length;
            return t('cluster.v2.staging.title')
                .replace('{done}', String(done))
                .replace('{total}', String(nodes.length));
        },

        stagingButtonLabel() {
            const missing = this.nodesMissingModel().length;
            return t(
                missing === 1
                    ? 'cluster.v2.staging.button_copying_one'
                    : 'cluster.v2.staging.button_copying',
            ).replace('{count}', String(missing));
        },

        // Phase 2 — the pre-staging activation path, unchanged.
        async postActivation(activation = this.activationRequestBody()) {
            if (!activation) {
                this.activateBusy = false;
                this.notify('error', 'The signed activation proposal is missing.');
                return;
            }
            try {
                await this.apiFetch(CLUSTER_V2_API.deployments, {
                    method: 'POST',
                    body: JSON.stringify(activation),
                });
                this.notify(
                    'success',
                    'Cluster activated. The distributed readiness check passed.',
                );
                this.stage = null;
                this.plan = null;
                this.planProposal = null;
                this.stagingJob = null;
                this.stagingActivation = null;
                this.membershipPanelOpen = false;
                this.membershipProposal = null;
                this.membershipError = '';
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                if (error?.status === 409) {
                    this.notify(
                        'warning',
                        'The plan changed since you reviewed it — rebuilding it now.',
                    );
                    this.stagingActivation = null;
                    await this.runPlan();
                } else {
                    this.notify(
                        'error',
                        error?.message || 'Activation failed',
                    );
                }
            } finally {
                this.stopStagingPoll();
                this.activateBusy = false;
            }
        },

        executionReplanBody(profile) {
            const deployment = this.configuredDeployment();
            const execution = this.deploymentExecution(deployment) || {};
            if (!deployment?.deployment_id) return null;
            const body = {
                deployment_id: deployment.deployment_id,
                execution_profile: profile,
                auto_tune: execution.auto_tune !== false,
                sampling_rank_only: execution.sampling_rank_only !== false,
                async_overlap: execution.async_overlap !== false,
                cache_affinity: execution.cache_affinity !== false,
                prompt_cache_ssd: execution.prompt_cache_ssd === true,
                prompt_cache_ssd_max_bytes:
                    Number(execution.prompt_cache_ssd_max_bytes) || 20 * (1024 ** 3),
                target_context_tokens:
                    Number(deployment.target_context_tokens) || 8192,
            };
            if (deployment.serving_mode === 'disaggregated') {
                body.serving_mode = 'disaggregated';
                body.prefill_rank = Number(deployment.prefill_rank);
                body.decode_rank = Number(deployment.decode_rank);
            }
            if (Number(execution.max_kv_size) > 0) {
                body.max_kv_size = Number(execution.max_kv_size);
            }
            if (Number(execution.ring_connections_per_ip) > 0) {
                body.ring_connections_per_ip = Number(
                    execution.ring_connections_per_ip,
                );
            }
            if (typeof deployment.mtp_enabled === 'boolean') {
                body.mtp_enabled = deployment.mtp_enabled;
            }
            if (Number(deployment.mtp_num_draft_tokens) > 0) {
                body.mtp_num_draft_tokens = Number(
                    deployment.mtp_num_draft_tokens,
                );
            }
            return body;
        },

        async previewExecutionProfile(profile) {
            const current = this.deploymentExecution()?.profile || 'balanced';
            if (profile === current) {
                this.executionReplan = null;
                return;
            }
            const body = this.executionReplanBody(profile);
            if (!body || this.executionReplanBusy) return;
            this.executionReplanBusy = true;
            try {
                const preview = await this.apiFetch(CLUSTER_V2_API.replan, {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
                const signature = preview?.plan?.placement_signature;
                if (typeof signature !== 'string' || signature.length < 16) {
                    throw new Error('The re-plan preview was not signed.');
                }
                this.executionReplan = { profile, body, preview };
            } catch (error) {
                this.executionReplan = null;
                this.notify(
                    'error',
                    error?.message || 'Could not preview the serving profile.',
                );
            } finally {
                this.executionReplanBusy = false;
            }
        },

        async applyExecutionProfileReplan() {
            const pending = this.executionReplan;
            const signature = pending?.preview?.plan?.placement_signature;
            if (!pending || !signature || this.executionReplanBusy) return;
            this.executionReplanBusy = true;
            try {
                await this.apiFetch(CLUSTER_V2_API.replan, {
                    method: 'POST',
                    body: JSON.stringify({
                        ...pending.body,
                        approved_placement: signature,
                    }),
                });
                this.executionProfile = pending.profile;
                this.executionReplan = null;
                this.notify(
                    'success',
                    'Serving profile applied and distributed readiness re-checked.',
                );
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message ||
                        'Could not apply the signed serving-profile re-plan.',
                );
            } finally {
                this.executionReplanBusy = false;
            }
        },

        // =================================================================
        // Active membership — pair first, then sign one N-node re-plan.
        // =================================================================
        deploymentMemberIds(deployment = this.configuredDeployment()) {
            return new Set(
                (deployment?.assignments || [])
                    .map((assignment) => assignment?.node_id)
                    .filter(Boolean),
            );
        },

        membershipCandidates(deployment = this.configuredDeployment()) {
            const active = this.deploymentMemberIds(deployment);
            return this.pairedDevices().filter(
                (device) => device?.node_id && !active.has(device.node_id),
            );
        },

        membershipActiveCount(deployment = this.configuredDeployment()) {
            return this.deploymentMemberIds(deployment).size;
        },

        toggleMembershipPanel() {
            const opening = !this.membershipPanelOpen;
            this.membershipPanelOpen = opening;
            this.membershipError = '';
            if (opening && !this.configuredDeployment()) {
                this.membershipReturnStage = this.stage;
                // Pairing must outrank the plan cursor while this panel is
                // open; the selected model/profile remain untouched.
                this.stage = null;
            }
            if (!opening) {
                this.membershipProposal = null;
                this.cancelPairing();
                if (!this.configuredDeployment()) {
                    this.stage =
                        this.membershipReturnStage ||
                        (this.selectedModelPath ? 'plan' : null);
                }
                this.membershipReturnStage = null;
            }
        },

        membershipRoleFor(nodeId, deployment = this.configuredDeployment()) {
            const assignment = (deployment?.assignments || []).find(
                (item) => item?.node_id === nodeId,
            );
            return assignment?.role || 'headless';
        },

        membershipPerformanceFor(
            nodeId,
            deployment = this.configuredDeployment(),
        ) {
            return (deployment?.performance_profiles || []).find(
                (profile) => profile?.node_id === nodeId,
            );
        },

        async previewMembershipExpansion() {
            const deployment = this.configuredDeployment();
            const candidates = this.membershipCandidates(deployment);
            if (
                !deployment?.deployment_id ||
                !candidates.length ||
                this.membershipBusy
            ) {
                return;
            }
            this.membershipBusy = true;
            this.membershipError = '';
            this.membershipProposal = null;
            try {
                await Promise.all(candidates.map((device) => this.probePeer(device)));
                const hosts = this.deploymentHosts();
                const roles = Object.fromEntries(
                    hosts.map((host) => [
                        host.node_id,
                        this.membershipRoleFor(host.node_id, deployment),
                    ]),
                );
                const measured = await this.apiFetch(
                    CLUSTER_V2_API.nodeBudgets,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            hosts: hosts.map((host) => ({
                                node_id: host.node_id,
                                ssh: host.ssh,
                                python_executable: host.python_executable,
                            })),
                            roles,
                        }),
                    },
                );
                const unavailable = (measured?.nodes || []).filter(
                    (node) => node?.unusable || !(Number(node?.capacity_bytes) > 0),
                );
                if (unavailable.length) {
                    throw new Error(
                        unavailable
                            .map(
                                (node) =>
                                    `${node.node_id}: ${node.error || 'memory probe unavailable'}`,
                            )
                            .join(' · '),
                    );
                }
                const nodes = (measured?.nodes || []).map((node) => {
                    const performance = this.membershipPerformanceFor(
                        node.node_id,
                        deployment,
                    );
                    return {
                        node_id: node.node_id,
                        capacity_bytes: Number(node.capacity_bytes),
                        reserve_bytes: Number(node.reserve_bytes || 0),
                        role: roles[node.node_id] || 'headless',
                        memory_guard_tier: 'balanced',
                        accelerator: 'metal',
                        ...(performance ? { performance } : {}),
                    };
                });
                const execution = this.deploymentExecution(deployment) || {};
                const proposal = await this.apiFetch(
                    CLUSTER_V2_API.autoconfigure,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            deployment_id: deployment.deployment_id,
                            model_path: deployment.model,
                            nodes,
                            hosts,
                            execution_profile: execution.profile || 'balanced',
                            prefer: 'speed',
                            strategy: 'auto',
                            detect_transports: true,
                            preflight: true,
                            auto_tune: execution.auto_tune !== false,
                            measure_performance: true,
                            sampling_rank_only:
                                execution.sampling_rank_only !== false,
                            async_overlap: execution.async_overlap !== false,
                            cache_affinity: execution.cache_affinity !== false,
                            prompt_cache_ssd:
                                execution.prompt_cache_ssd === true,
                            prompt_cache_ssd_max_bytes:
                                Number(execution.prompt_cache_ssd_max_bytes) ||
                                20 * (1024 ** 3),
                            max_kv_size:
                                Number(execution.max_kv_size) || undefined,
                            ring_connections_per_ip:
                                Number(execution.ring_connections_per_ip) ||
                                undefined,
                            target_context_tokens:
                                Number(deployment.target_context_tokens) || 8192,
                            mtp_enabled: deployment.mtp_enabled === true,
                            mtp_num_draft_tokens:
                                Number(deployment.mtp_num_draft_tokens) || null,
                        }),
                    },
                );
                this.membershipProposal = proposal;
                if (!proposal?.ready_to_activate) {
                    this.membershipError =
                        proposal?.fabric_blocker ||
                        proposal?.preflight ||
                        'The expanded pool is not ready to activate.';
                }
            } catch (error) {
                this.membershipError =
                    error?.message || 'Could not preview the expanded cluster.';
            } finally {
                this.membershipBusy = false;
            }
        },

        membershipPlanAssignments() {
            const assignments = this.membershipProposal?.plan?.assignments;
            return Array.isArray(assignments) ? assignments : [];
        },

        membershipPlanNodeName(assignment) {
            const device = this.allDevices().find(
                (item) => item?.node_id === assignment?.node_id,
            );
            return device
                ? this.deviceName(device)
                : assignment?.node_id || `Rank ${assignment?.rank ?? '?'}`;
        },

        membershipPlanDetail(assignment) {
            const gib = Number(assignment?.planned_weight_bytes || 0) / (1024 ** 3);
            const tp = Number(assignment?.tensor_parallel_size || 1);
            return tp > 1
                ? `Tensor rank ${Number(assignment?.tensor_parallel_rank || 0) + 1}/${tp} · ${gib.toFixed(1)} GiB planned`
                : `Layers ${assignment?.start_layer ?? 0}–${assignment?.end_layer ?? 0} · ${gib.toFixed(1)} GiB planned`;
        },

        async applyMembershipExpansion() {
            const proposal = this.membershipProposal;
            const activation = proposal?.activation;
            if (
                !proposal?.ready_to_activate ||
                !activation ||
                this.membershipBusy ||
                this.activateBusy
            ) {
                return;
            }
            this.membershipBusy = true;
            this.activateBusy = true;
            try {
                await this.stageModelToPeers(activation);
            } finally {
                this.membershipBusy = false;
            }
        },

        hydratePlannerFromDeployment(deployment) {
            const execution = this.deploymentExecution(deployment) || {};
            this.executionProfile = String(
                execution.profile || deployment?.execution?.profile || 'balanced',
            );
            this.promptCacheSsd = Boolean(
                execution.prompt_cache_ssd ?? deployment?.prompt_cache_ssd,
            );
            this.promptCacheSsdMaxGiB = Math.max(
                1,
                Math.round(
                    Number(
                        execution.prompt_cache_ssd_max_bytes ||
                            deployment?.prompt_cache_ssd_max_bytes ||
                            20 * 1024 ** 3,
                    ) /
                        1024 ** 3,
                ),
            );
            this.targetContextTokens = Math.max(
                1,
                Number(deployment?.target_context_tokens || 32768),
            );
            this.nodeRoles = Object.fromEntries(
                (deployment?.assignments || [])
                    .filter((assignment) => assignment?.node_id && assignment?.role)
                    .map((assignment) => [assignment.node_id, assignment.role]),
            );
            // A different architecture may need a different split. Automatic
            // retains all the current serving/cache/memory choices while
            // letting the catalogue select TP or pipeline for the new model.
            this.planStrategy = 'auto';
        },

        resetModelPicker() {
            this.selectedModelPath = '';
            this.modelSearch = '';
            this.modelOptions = [];
            this.modelsError = '';
            this.catalogueModels = null;
            this.catalogueFailed = false;
            this.plan = null;
            this.planProposal = null;
            this.planError = '';
            this.planFitFailure = null;
            this.checks.benchmark = null;
            this.stagingJob = null;
            this.stagingActivation = null;
            this.stagingError = '';
        },

        async unloadDeploymentWeights(deployment) {
            const id = deployment?.deployment_id;
            if (!id || this.clusterLifecycleBusy) return;
            if (this.confirmUnloadFor !== id) {
                this.confirmUnloadFor = id;
                this.confirmChangeModelFor = '';
                return;
            }
            this.confirmUnloadFor = '';
            this.clusterLifecycleBusy = true;
            this.hydratePlannerFromDeployment(deployment);
            try {
                await this.apiFetch(CLUSTER_V2_API.deploymentUnload(id), {
                    method: 'POST',
                });
                this.notify(
                    'success',
                    'Cluster weights unloaded. The signed setup is still ready to load again.',
                );
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message || 'Could not unload the cluster weights.',
                );
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        async loadDeploymentWeights(deployment) {
            const id = deployment?.deployment_id;
            if (!id || this.clusterLifecycleBusy) return;
            this.clusterLifecycleBusy = true;
            try {
                await this.apiFetch(CLUSTER_V2_API.deploymentLoad(id), {
                    method: 'POST',
                });
                this.notify(
                    'success',
                    'Cluster weights loaded and every rank passed readiness.',
                );
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message || 'Could not load the cluster weights.',
                );
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        beginModelChange(deployment) {
            const id = deployment?.deployment_id;
            if (!id || this.clusterLifecycleBusy) return;
            this.confirmChangeModelFor = id;
            this.confirmUnloadFor = '';
            this.confirmDeactivateFor = '';
        },

        cancelModelChange() {
            this.confirmChangeModelFor = '';
        },

        async changeClusterModel(deployment) {
            const id = deployment?.deployment_id;
            if (
                !id ||
                this.confirmChangeModelFor !== id ||
                this.clusterLifecycleBusy
            ) {
                return;
            }
            this.clusterLifecycleBusy = true;
            this.hydratePlannerFromDeployment(deployment);
            try {
                await this.apiFetch(CLUSTER_V2_API.deployment(id), {
                    method: 'DELETE',
                });
                this.confirmChangeModelFor = '';
                this.resetModelPicker();
                await this.refreshDeployments();
                await this.refreshRuntime();
                this.enterPlan();
                this.notify(
                    'info',
                    'Previous shards unloaded. Choose the next model; current serving and memory settings are prefilled.',
                );
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message || 'Could not switch the clustered model.',
                );
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        async deactivateDeployment(deployment) {
            const id = deployment?.deployment_id;
            if (!id || this.clusterLifecycleBusy) return;
            if (this.confirmDeactivateFor !== id) {
                this.confirmDeactivateFor = id;
                return;
            }
            this.confirmDeactivateFor = '';
            this.confirmChangeModelFor = '';
            this.confirmUnloadFor = '';
            this.clusterLifecycleBusy = true;
            try {
                await this.apiFetch(CLUSTER_V2_API.deployment(id), {
                    method: 'DELETE',
                });
                this.notify('info', 'Cluster deactivated.');
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not deactivate',
                );
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        // =====================================================================
        // Feedback — toasts, never modals
        // =====================================================================
        notify(type, message, timeoutMs = 6000) {
            const id = ++this.toastSeq;
            this.toasts.push({ id, type, message });
            setTimeout(() => {
                this.toasts = this.toasts.filter((toast) => toast.id !== id);
            }, timeoutMs);
        },

        dismissToast(id) {
            this.toasts = this.toasts.filter((toast) => toast.id !== id);
        },

        toastTone(type) {
            return {
                success: 'border-green-200 bg-green-50 text-green-800',
                error: 'border-red-200 bg-red-50 text-red-800',
                warning: 'border-amber-200 bg-amber-50 text-amber-800',
                info: 'border-neutral-200 bg-white text-neutral-800',
            }[type] || 'border-neutral-200 bg-white text-neutral-800';
        },

        toastIcon(type) {
            return {
                success: 'check-circle-2',
                error: 'circle-alert',
                warning: 'triangle-alert',
                info: 'info',
            }[type] || 'info';
        },

        retryConnection() {
            this.devicesUnreachable = false;
            this.devicesFailureCount = 0;
            this.refreshDevices();
        },

        copyInstallCommand() {
            const command = 'brew install jundot/omlx/omlx';
            const done = () => {
                this.installCommandCopied = true;
                setTimeout(() => (this.installCommandCopied = false), 2000);
            };
            if (navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(command).then(done, done);
            } else {
                done();
            }
        },
    };
}
