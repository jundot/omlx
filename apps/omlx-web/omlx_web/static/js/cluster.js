// oMLX Cluster wizard — Alpine.js component backing
// omlx_web/templates/dashboard/_cluster.html.
//
// Contract (Module C): consume ONLY the Module A/B endpoints plus the
// planner/activate API:
//
//   Module A/B:
//     GET    /api/cluster/devices               — {paired, discovered, self}, polled at 1 Hz
//     POST   /api/cluster/pair/approve          — {node_id, code}
//     POST   /api/cluster/pair/deny             — {node_id}
//     POST   /api/cluster/pair/join             — {coordinator_addr} — joiner: mint + show the code
//     GET    /api/cluster/pair/join             — local join snapshot, polled at 1 Hz (drives approval)
//     POST   /api/cluster/pair/join/cancel      — abandon the join in progress
//     POST   /api/cluster/devices/manual        — {ip, port} seed + probe a peer by address
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
//
// State machine (wizardState()): empty → discovering → device_card → pairing →
// checks → plan → active, with error as an overlay state (banner + toasts,
// never a modal). N device cards are rendered; there is no 2-Mac cap.
//
// This file is loaded as a plain script before Alpine initializes, exactly
// like dashboard.js, so `clusterWizard` is a global factory referenced from
// x-data in _cluster.html.

function clusterWizard() {
    const CLUSTER_API = {
        devices: '/api/cluster/devices',
        discoveryHealth: '/api/cluster/discovery/health',
        pairApprove: '/api/cluster/pair/approve',
        pairDeny: '/api/cluster/pair/deny',
        pairJoin: '/api/cluster/pair/join',
        pairJoinCancel: '/api/cluster/pair/join/cancel',
        pairJoinCleanup: '/api/cluster/pair/join/cleanup',
        manualDevice: '/api/cluster/devices/manual',
        sshUser: (nodeId) => `/api/cluster/devices/${encodeURIComponent(nodeId)}/ssh-user`,
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
        deployment: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}`,
        deploymentLoad: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}/load`,
        deploymentUnload: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}/unload`,
    };

    // The Cluster Assistant surface: five admin routes, and nothing else.
    // the first UI slice notes: the browser never launches, stages, deploys or
    // starts inference itself — it calls these and renders the answer.
    // A 32 GB Mac cannot spare the 32 GiB workstation reserve and still hold a
    // pipeline stage; the assistant treats it as Headless (see assistantRoles).
    const ASSISTANT_SMALL_MAC_RAM_GB = 64;
    // The backend's recommend route accepts at most this many candidates
    // (models.py MAX_RECOMMENDATION_CANDIDATES). A larger catalogue is narrowed
    // client-side, never sent and refused.
    const ASSISTANT_MAX_CANDIDATES = 32;
    // A minutes-long measurement must survive a transient network blip:
    // tolerate this many consecutive poll errors (with backoff) before failing.
    const ASSISTANT_POLL_MAX_ERRORS = 3;

    const ASSISTANT_API = {
        measure: '/admin/api/cluster/assistant/measure',
        status: '/admin/api/cluster/assistant/status',
        recommend: '/admin/api/cluster/assistant/recommend',
        plan: '/admin/api/cluster/assistant/plan',
        // The free-space relay (the live-run notes): one model, on one measured
        // Mac, deleted by that Mac's own route. The browser asks; it never
        // reaches a disk or an SSH target itself.
        models: (nodeId, modelId) =>
            `/admin/api/cluster/assistant/models/${encodeURIComponent(nodeId)}/${encodeURIComponent(modelId)}`,
    };

    // exo-style data layer: one snapshot endpoint polled once per second
    // while the tab is visible; the UI is a pure function of the snapshot.
    const CLUSTER_POLL_MS = 1000;
    const CLUSTER_DEPLOYMENTS_EVERY_TICKS = 5;
    // A missing cluster backend (404) flips the error state immediately; flaky
    // networks get a few grace failures first.
    const CLUSTER_FAILURE_GRACE = 3;

    const CLUSTER_LINK_META = {
        tb: { label: window.t('cluster.link.thunderbolt'), icon: 'zap' },
        ethernet: { label: window.t('cluster.link.ethernet'), icon: 'cable' },
        wifi: { label: window.t('cluster.link.wifi'), icon: 'wifi' },
        // This is the address used to find/control the peer. It is not the
        // collective fabric selected by the signed deployment.
        tailscale: { label: window.t('cluster.link.tailscale_control'), icon: 'globe' },
        unknown: { label: window.t('cluster.link.network'), icon: 'help-circle' },
    };

    // Offline mirror of omlx/cluster/node_role.py (NodeRole.reserve_for):
    // a workstation keeps max(32 GiB, 50%) of its Mac, a headless node keeps
    // 10%. The wizard prefers GET /admin/api/cluster/node-roles, which exposes
    // these same numbers from the server; this mirror exists only so the
    // usable-budget labels still render when that endpoint is unreachable.
    // If node_role.py changes, this mirror must change with it.
    const CLUSTER_ROLE_FALLBACK = {
        workstation: {
            key: 'workstation',
            label: window.t('cluster.role.workstation'),
            reserve_bytes: 32 * 1024 ** 3,
            reserve_fraction: 0.5,
        },
        headless: {
            key: 'headless',
            label: window.t('cluster.role.headless'),
            reserve_bytes: 0,
            reserve_fraction: 0.1,
        },
    };

    const t = (key) => window.t(key);

    return {
        // ---- snapshot state -------------------------------------------------
        devicesPayload: null,
        sshUserDrafts: {},
        sshUserSaving: {},
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
            cleanup_pending: false,
            busy: false,
            target_name: '',
        },
        joinApprovedNotified: false,
        joinDeniedNotified: false,
        joinRevision: 0,
        confirmForgetCleanup: false,

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
        },
        // Prefill Accelerator checks for one paired Mac (GET /check).
        accelerator: { nodeId: null, running: false, result: null, error: '', copies: null, deleting: [] },
        acceleratorRestored: false,
        // On a helper Mac: its running helper and the Mac that set it up.
        helperRole: null,
        // A Mac that pressed Pair on this one: {node_id, name}.
        pairInvite: null,
        // The chooser opened by itself after pairing, not from Change setup.
        purposeAuto: false,

        // ---- assistant (the minimal Screen-3 slice) --------------------------
        // One card inside the plan section that drives measure → recommend →
        // plan through the assistant admin API. It renders the backend's
        // answer; it never derives a plan, a rank order, a coordinator or an
        // admission, and it never launches anything itself.
        assistant: {
            // idle → measuring → recommending → measuring_model → planning,
            // plus failed.
            state: 'idle',
            // Which measurement is being taken: the entry measurement (then
            // recommend) or the model-bound one for the recommended model.
            phase: 'recommend',
            busy: false,
            generation: 0,
            measurementId: '',
            measurementModel: '',
            recommendation: null,
            selectedModel: '',
            selectedModelId: '',
            selectedModelLabel: '',
            selectedContextTokens: null,
            // Phase A: "minimal" uses the fewest Macs that fit; "all" is the
            // advanced override. The solver still shows every lever.
            nodeSelection: 'minimal',
            // The user explicitly accepted the offered 2-Mac fallback after a
            // topology_incomplete refusal. It is part of the plan signature.
            confirmTwoMacFallback: false,
            topologyFallback: null,
            missingLinks: [],
            plan: null,
            // Frozen plan-time inputs: the exact object sent to plan.
            inputs: null,
            // The measured Macs from the last status poll, with each Mac's own
            // model list. Only the Mac the user picks is ever listed for delete.
            measurementNodes: [],
            freeSpaceOpen: false,
            freeSpaceNode: '',
            freeSpaceConfirm: '',
            freeSpaceBusy: false,
            freeSpaceError: '',
            warnings: [],
            error: null,
            measureTimer: null,
            measurePollBusy: false,
            // Bounded retry on transient poll errors: consecutive failures and
            // the ticks to skip before the next attempt (1, 2, 4 … backoff).
            measurePollErrors: 0,
            measurePollSkip: 0,
            copied: '',
        },

        // ---- plan ------------------------------------------------------------
        modelOptions: [],
        modelsLoading: false,
        modelsError: '',
        selectedModelPath: '',
        modelSearch: '',
        // Execution strategy for the split: 'auto' | 'tensor' | 'pipeline'.
        // The server resolves its TP degree, host order and backend together;
        // the client never reconstructs those decisions after preview.
        planStrategy: 'auto',
        // Scheduler limits are selected as one of the server-owned execution
        // profiles. The worker may safely reduce these values for available
        // headroom, and the active view always renders those resolved values.
        // Continuous batching itself is automatic for batchable models; this
        // profile controls its target width and prompt/decode admission limits.
        executionProfile: 'balanced',
        // Match the server default for new clusters. Existing deployments keep
        // their explicit choice when hydrated, and the SSD quota stays bounded.
        promptCacheSsd: true,
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
        confirmDeactivateFor: '',
        confirmForgetFor: '',
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

        // ---- bounded advanced tools (the old v1 console is removed) ---------
        advancedOpen: false,
        diagnosticsLoading: false,
        diagnosticsError: '',
        cudaJoin: {
            controllerIp: '',
            loading: false,
            command: '',
            joinId: '',
            expiresAt: 0,
            copied: false,
            error: '',
            status: { join_keys: [], nodes: [] },
        },
        cudaFabricLoading: false,
        cudaFabricError: '',
        cudaFabricResult: null,
        cudaFabricMemberA: '',
        cudaFabricMemberB: '',
        rdmaLinks: { loading: false, verifying: '', error: '', data: null },

        // ---- feedback ----------------------------------------------------------
        toasts: [],
        toastSeq: 0,
        installCommandCopied: false,

        pollTimer: null,
        tickBusy: false,
        runtimeRevision: 0,
        choosingModel: false,
        stagingPurpose: 'plan',
        stagingGeneration: 0,
        stagingPollBusy: false,
        stagingClaimed: false,
        selectedDeploymentId: '',
        measuredNodeBudgets: {},
        tickCount: 0,

        // =====================================================================
        // Lifecycle
        // =====================================================================
        init() {
            if (this.pollTimer !== null) return;
            this.tick();
            this.pollTimer = setInterval(() => this.tick(), CLUSTER_POLL_MS);
        },

        wizardVisible() {
            return this.mainTab === 'cluster' && !document.hidden;
        },

        async tick() {
            if (!this.wizardVisible() || this.tickBusy) return;
            this.tickBusy = true;
            try {
                await this.refreshDevices();
                await this.refreshHelperRole();
                await this.refreshJoinState();
                await this.refreshInvite();
                await this.refreshRuntime();
                this.tickCount += 1;
                if (
                    !this.deploymentsLoaded ||
                    this.tickCount % CLUSTER_DEPLOYMENTS_EVERY_TICKS === 0
                ) {
                    await this.refreshDeployments();
                    await this.refreshRdmaLinks();
                }
                await this.restoreAccelerator();
            } finally {
                this.tickBusy = false;
            }
        },

        destroy() {
            if (this.pollTimer !== null) clearInterval(this.pollTimer);
            this.pollTimer = null;
            this.runtimeRevision += 1;
            this.assistant.generation += 1;
            this.stopAssistantPolls();
            this.dismissStaging();
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
                throw new Error(window.t('cluster.err.sign_in_required'));
            }
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                let payload = null;
                try {
                    payload = await response.json();
                    if (payload && payload.detail) {
                        detail =
                            typeof payload.detail === 'string'
                                ? payload.detail
                                : JSON.stringify(payload.detail);
                    } else if (payload && payload.message) {
                        // The assistant routes refuse with a flat
                        // {code, message, action} body (no FastAPI `detail`),
                        // and the UI renders that message and action verbatim.
                        detail = String(payload.message);
                    }
                } catch (ignored) {
                    /* non-JSON error body */
                }
                const error = new Error(detail);
                error.status = response.status;
                // Keep the machine-readable refusal so the assistant card can
                // render error.code / error.message / error.action without
                // parsing prose.
                error.payload = payload;
                throw error;
            }
            if (response.status === 204) return null;
            return response.json();
        },

        async refreshDevices() {
            try {
                const payload = await this.apiFetch(CLUSTER_API.devices);
                this.devicesPayload = payload || {};
                this.devicesLoaded = true;
                this.devicesError = '';
                this.devicesFailureCount = 0;
                this.devicesUnreachable = false;
            } catch (error) {
                this.devicesFailureCount += 1;
                this.devicesError = error?.message || window.t('cluster.err.api_unreachable');
                // 404 means the discovery backend is not serving — that is
                // a hard, actionable failure, not network noise.
                if (
                    error?.status === 404 ||
                    this.devicesFailureCount >= CLUSTER_FAILURE_GRACE
                ) {
                    this.devicesUnreachable = true;
                }
            }
        },

        async refreshDeployments() {
            try {
                const payload = await this.apiFetch(CLUSTER_API.deployments);
                this.deploymentsPayload = payload?.deployments || [];
                this.deploymentsLoaded = true;
                this.ensureColdModelSelection();
            } catch (error) {
                // Deployments are the pre-existing API; a failure here should
                // not tear down the discovery UI, just surface a toast once.
                if (this.deploymentsLoaded) {
                    this.notify(
                        'warning',
                        error?.message || window.t('cluster.err.refresh_deployments'),
                    );
                }
            }
        },

        async refreshRuntime() {
            const revision = ++this.runtimeRevision;
            try {
                const payload = await this.apiFetch(CLUSTER_API.runtime);
                if (revision !== this.runtimeRevision) return;
                this.runtimePayload = payload || { jobs: [], launchers: [] };
                this.runtimeLoaded = true;
                this.runtimeError = '';
                this.ensureColdModelSelection();
            } catch (error) {
                if (revision !== this.runtimeRevision) return;
                // Fail closed. A previous ready snapshot must never remain green
                // when the ownership endpoint can no longer prove residency.
                this.runtimePayload = null;
                this.runtimeLoaded = false;
                this.runtimeError =
                    error?.message || window.t('cluster.err.runtime_unavailable');
            }
        },

        async refreshDiscoveryHealth() {
            try {
                this.discoveryHealth = await this.apiFetch(
                    CLUSTER_API.discoveryHealth,
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
            if (this.join.busy) return;
            const revision = this.joinRevision;
            try {
                const snapshot = await this.apiFetch(CLUSTER_API.pairJoin);
                if (!snapshot || revision !== this.joinRevision) return;
                const previous = this.join.state;
                this.join = { ...this.join, ...snapshot, busy: false };
                if (!snapshot.cleanup_pending) this.confirmForgetCleanup = false;
                if (
                    snapshot.state === 'approved' &&
                    !this.joinApprovedNotified
                ) {
                    this.joinApprovedNotified = true;
                    this.notify(
                        'success',
                        window.t('cluster.toast.joined_cluster').replace('{name}', this.joinTargetName()),
                    );
                    await this.refreshDevices();
                    this.choosePurpose(true);
                } else if (
                    snapshot.state === 'denied' &&
                    previous !== 'denied' &&
                    !this.joinDeniedNotified
                ) {
                    this.joinDeniedNotified = true;
                    this.notify(
                        'error',
                        window.t('cluster.toast.join_denied').replace('{name}', this.joinTargetName()),
                    );
                    // Denied is terminal server-side; reset locally so the
                    // panel clears instead of sticking on the refusal.
                    await this.cancelJoin({ silent: true });
                }
            } catch (error) {
                if (revision !== this.joinRevision) return;
                // A 404 means this backend predates the joiner endpoints —
                // stay idle rather than tearing down the rest of the wizard.
                if (error?.status !== 404) {
                    this.join = {
                        ...this.join,
                        error: error?.message || window.t('cluster.err.join_status_unavailable'),
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
            // N device cards; there is no 2-Mac cap.
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
            const selected = deployments.find((item) => item.deployment_id === this.selectedDeploymentId);
            if (selected) return selected;

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
            // Keep saved setups manageable after unload/failure. Residency is
            // reported separately by deploymentRuntimeState/activeDeployment.
            return loading || deployments[0];
        },

        ensureColdModelSelection() {
            if (
                this.deploymentsLoaded && this.selectedDeploymentId &&
                !this.deploymentsPayload.some((item) => item.deployment_id === this.selectedDeploymentId)
            ) {
                this.selectedDeploymentId = '';
            }
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
                    label: window.t('cluster.batch.label_unavailable'),
                    detail:
                        capability.reason ||
                        window.t('cluster.batch.unavailable_reason'),
                    tone: 'bg-amber-50 border-amber-200 text-amber-700',
                    target,
                };
            }
            if (target <= 1 || capability?.enabled === false) {
                return {
                    label: window.t('cluster.batch.label_sequential'),
                    detail: window.t('cluster.batch.sequential_detail'),
                    tone: 'bg-neutral-50 border-neutral-200 text-neutral-600',
                    target,
                };
            }
            if (lastSize > 1) {
                return {
                    label: window.t('cluster.batch.label_batched').replace('{n}', String(lastSize)),
                    detail: window.t('cluster.batch.batched_detail').replace('{last}', String(lastSize)).replace('{target}', String(target)),
                    tone: 'bg-green-50 border-green-200 text-green-700',
                    target,
                };
            }
            return {
                label: window.t('cluster.batch.label_automatic'),
                detail: window.t('cluster.batch.automatic_detail').replace('{target}', String(target)),
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
                return `${window.t('cluster.req.active_count').replace('{n}', String(active))}${hidden ? ` · ${window.t('cluster.req.not_shown').replace('{n}', String(hidden))}` : ''}`;
            }
            return metrics?.last_request ? window.t('cluster.req.last_completed') : window.t('cluster.req.waiting');
        },

        requestPhaseLabel(request) {
            // Stable, untranslated identifier. The display site localises it
            // via window.t('cluster.phase.<id>'); requestPhaseTone() switches
            // on the same identifiers, so the two never drift apart.
            if (request?._history) {
                return request?.status === 'failed' ? 'failed' : 'complete';
            }
            if (request?.prefill_progress?.active) return 'prefill';
            if (Number(request?.completion_tokens || 0) > 0) return 'decode';
            return 'queued';
        },

        requestPhaseTone(request) {
            const phase = this.requestPhaseLabel(request);
            if (phase === 'prefill') return 'bg-blue-50 border-blue-200 text-blue-700';
            if (phase === 'decode') return 'bg-green-50 border-green-200 text-green-700';
            if (phase === 'failed') return 'bg-red-50 border-red-200 text-red-700';
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
                return window.t('cluster.req.cache_hit');
            }
            // The last progress sample freezes at the compute boundary. Keep
            // its average after decode begins; TTFT also includes queueing and
            // can make a fast prefill look artificially slow under contention.
            const rate = Number(
                progress?.average_speed || request?.prefill_tps || 0,
            );
            return this.formatRequestRate(rate);
        },

        requestPrefillDetail(request) {
            if (request?.prefill_progress?.active) return window.t('cluster.req.live_average');
            const prompt = Math.max(0, Number(request?.prompt_tokens || 0));
            const cached = Math.min(
                prompt,
                Math.max(0, Number(request?.cached_tokens || 0)),
            );
            if (cached > 0) {
                return window.t('cluster.req.prefill_reuse')
                .replace('{cached}', cached.toLocaleString())
                .replace('{new}', Math.max(0, prompt - cached).toLocaleString());
            }
            return window.t('cluster.req.prompt_average');
        },

        requestDecodeRate(request) {
            const tokens = Math.max(0, Number(request?.completion_tokens || 0));
            if (!request?._history && tokens === 1) return window.t('cluster.req.measuring');
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
                return window.t('cluster.req.prefill_progress').replace('{processed}', processed.toLocaleString()).replace('{total}', total.toLocaleString());
            }
            const uncached = Math.max(0, prompt - cached);
            return `${window.t('cluster.req.new_tokens').replace('{n}', uncached.toLocaleString())}${cached ? ` · ${window.t('cluster.req.cached_tokens').replace('{n}', cached.toLocaleString())}` : ''}`;
        },

        requestDecodeDetail(request) {
            const tokens = Math.max(0, Number(request?.completion_tokens || 0));
            return window.t('cluster.req.generated').replace('{n}', tokens.toLocaleString());
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
                : window.t('cluster.deploy.worker_stopped');
        },

        deploymentStatus(deployment = this.configuredDeployment()) {
            const state = this.deploymentRuntimeState(deployment);
            if (state === 'ready') {
                return {
                    state,
                    eyebrow: window.t('cluster.deploy.status_running'),
                    label: window.t('cluster.deploy.status_ready'),
                    detail:
                        window.t('cluster.deploy.status_running_detail'),
                    tone: 'bg-green-50 border-green-200 text-green-700',
                    pulse: true,
                };
            }
            if (state === 'loading') {
                return {
                    state,
                    eyebrow: window.t('cluster.deploy.status_starting'),
                    label: window.t('cluster.deploy.status_loading'),
                    detail:
                        window.t('cluster.deploy.status_starting_detail'),
                    tone: 'bg-blue-50 border-blue-200 text-blue-700',
                    pulse: true,
                };
            }
            if (state === 'failed') {
                return {
                    state,
                    eyebrow: window.t('cluster.deploy.status_attention'),
                    label: window.t('cluster.deploy.status_failed'),
                    detail: this.deploymentFailureReason(deployment),
                    tone: 'bg-red-50 border-red-200 text-red-700',
                    pulse: false,
                };
            }
            if (state === 'unknown') {
                return {
                    state,
                    eyebrow: window.t('cluster.deploy.status_configured'),
                    label: window.t('cluster.deploy.status_checking'),
                    detail: this.runtimeError
                        ? window.t('cluster.deploy.status_runtime_unavailable').replace('{message}', this.runtimeError)
                        : window.t('cluster.deploy.status_checking_detail'),
                    tone: 'bg-neutral-50 border-neutral-200 text-neutral-600',
                    pulse: true,
                };
            }
            return {
                state: 'configured',
                eyebrow: window.t('cluster.deploy.status_configured'),
                label: window.t('cluster.deploy.status_not_loaded'),
                detail: window.t('cluster.deploy.status_not_loaded_detail'),
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
                ? window.t('cluster.deploy.cache_memory_ssd').replace(
                      '{gib}',
                      String(Math.round(Number(deployment?.execution?.prompt_cache_ssd_max_bytes || 20 * (1024 ** 3)) / (1024 ** 3))),
                  )
                : window.t('cluster.deploy.cache_memory_only');
        },

        // =====================================================================
        // State machine — empty / discovering / device_card / pairing / checks /
        // plan / active / error
        // =====================================================================
        wizardState() {
            if (this.devicesUnreachable) return 'error';
            // A durable deployment keeps its management panel mounted, but its
            // badge is driven by deploymentRuntimeState(), not by persistence.
            // An explicit model change (choosingModel) still reopens the picker
            // even when a deployment exists — the shortcut must not hide it.
            if (this.configuredDeployment() && !this.choosingModel) return 'active';
            // An incoming join request (or an open pairing form / an active
            // local join) must show its controls immediately, even when the
            // wizard already advanced to checks/plan/purpose. Otherwise only
            // the device label changes and the code input stays hidden until
            // a reload resets `stage`.
            if (
                this.pairing.target ||
                this.pendingApprovals().length ||
                this.joinActive()
            ) {
                return 'pairing';
            }
            if (this.choosingModel && this.stage === 'plan') return 'plan';
            if (this.configuredDeployment()) return 'active';
            if (this.stage === 'plan' && this.pairedDevices().length) {
                return 'plan';
            }
            if (this.stage === 'checks' && this.pairedDevices().length) {
                return 'checks';
            }
            if (this.stage === 'purpose' && this.purposeAuto && this.helperRoleHost()) {
                return 'helper';
            }
            if (
                (this.stage === 'purpose' || this.stage === 'accelerator') &&
                this.pairedDevices().length
            ) {
                return this.stage;
            }
            if (this.stage === null && this.helperRoleHost()) return 'helper';
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
                { key: 'discover', title: window.t('cluster.steps.discover_title'), hint: window.t('cluster.steps.discover_hint') },
                { key: 'pair', title: window.t('cluster.steps.pair_title'), hint: window.t('cluster.steps.pair_hint') },
                { key: 'checks', title: window.t('cluster.steps.checks_title'), hint: window.t('cluster.steps.checks_hint') },
                { key: 'plan', title: window.t('cluster.steps.plan_title'), hint: this.planStepHint() },
                { key: 'active', title: window.t('cluster.steps.active_title'), hint: window.t('cluster.steps.active_hint') },
            ];
            if (wizard === 'accelerator' || wizard === 'helper') {
                // Prefill Accelerator ends at its checks; nothing is split or activated.
                const ready =
                    wizard === 'helper' || (!this.accelerator.running && this.acceleratorReady());
                steps[2].hint = window.t('cluster.steps.accelerator_checks_hint');
                return steps.slice(0, 3).map((step, index) => ({
                    ...step,
                    state: index < 2 || ready ? 'done' : 'active',
                }));
            }
            // Map the 8 UI states onto the 5 step slots.
            const slotFor = {
                empty: 0,
                discovering: 0,
                device_card: 1,
                pairing: 1,
                purpose: 2,
                accelerator: 2,
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
                staging: window.t('cluster.progress.staging'),
                starting: window.t('cluster.progress.starting'),
                preflight: window.t('cluster.progress.preflight'),
                initializing: window.t('cluster.progress.initializing'),
                initializing_full_replica: window.t('cluster.progress.initializing_full_replica'),
                loading_weights: window.t('cluster.progress.loading_weights'),
                materializing_fixed: window.t('cluster.progress.materializing_fixed'),
                materializing_layers: window.t('cluster.progress.materializing_layers'),
                tensor_ready: window.t('cluster.progress.tensor_ready'),
                weights_resident: window.t('cluster.progress.weights_resident'),
                validating: window.t('cluster.progress.validating'),
                warming_prefill_shape: window.t('cluster.progress.warming_prefill_shape'),
                ready: window.t('cluster.progress.ready'),
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
                return window.t('cluster.progress.preparing_launcher');
            }
            const ready = jobs.filter(
                (job) => job?.phase === 'ready' && job?.live === true,
            ).length;
            const expected = Math.max(
                jobs.length,
                ...jobs.map((job) => Math.max(0, Number(job?.world_size || 0))),
            );
            return window.t('cluster.progress.ranks_ready').replace('{ready}', String(ready)).replace('{expected}', String(expected));
        },

        // Step-4 hint is strategy-aware: a tensor split gives every Mac every
        // layer, so "Layers per Mac" would describe the wrong thing.
        planStepHint() {
            return this.planStrategy === 'tensor'
                ? t('cluster.steps.plan_hint_tensor')
                : window.t('cluster.steps.plan_hint_pipeline');
        },

        // =====================================================================
        // Version parity — actionable banner, device stays visible
        // =====================================================================
        versionMismatches() {
            const self = this.selfDevice();
            if (!self || !self.version) return [];
            // A helper daemon has its own version number.
            return this.splitDevices()
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
                window.t('cluster.device.unknown')
            );
        },

        deviceRamGb(device) {
            const ram = device?.caps?.ram_gb;
            return typeof ram === 'number' && ram > 0 ? ram : null;
        },

        deviceRamLabel(device) {
            const ram = this.deviceRamGb(device);
            return ram ? `${ram} GB` : window.t('cluster.device.memory_unknown');
        },

        deviceChipLabel(device) {
            return (
                device?.caps?.chip ||
                window.t(
                    this.deviceHelperOnly(device)
                        ? 'cluster.device.cuda_helper'
                        : 'cluster.device.apple_silicon',
                )
            );
        },

        // A helper daemon is only a Prefill Accelerator: no SSH, no split.
        deviceHelperOnly(device) {
            return device?.caps?.control === 'agent';
        },

        deviceLoopbackOnly(device) {
            return !!device?.caps?.loopback_only;
        },

        deviceLinkMeta(device) {
            return (
                CLUSTER_LINK_META[device?.link] ||
                CLUSTER_LINK_META.unknown
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
            if (device?.is_self) return window.t('cluster.device.this_mac');
            if (device?.state === 'awaiting_approval') return window.t('cluster.device.wants_to_join');
            if (device?.paired) return window.t('cluster.device.paired');
            if (device?.state === 'dead') return window.t('cluster.device.unreachable');
            if (device?.state === 'suspect') return window.t('cluster.device.connection_shaky');
            return window.t('cluster.device.found_nearby');
        },

        combinedMemoryLabel() {
            const total = this.allDevices()
                .filter((device) => !this.deviceHelperOnly(device))
                .reduce((sum, device) => sum + (this.deviceRamGb(device) || 0), 0);
            return total ? window.t('cluster.device.combined').replace('{n}', String(total)) : '';
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
            // That Mac then asks its user to show the code; best effort.
            this.apiFetch('/api/cluster/pair/invite', {
                method: 'POST',
                body: JSON.stringify({ node_id: device.node_id }),
            }).catch(() => {});
        },

        pairingStepOne(device = this.pairing.target) {
            // A helper daemon has no screen; its code is in its log.
            const key = this.deviceHelperOnly(device)
                ? 'cluster.pairing.step1_helper'
                : 'cluster.pairing.step1';
            return window.t(key)
                .replaceAll('{name}', this.deviceName(device))
                .replaceAll('{self}', this.deviceName(this.selfDevice()));
        },

        async refreshInvite() {
            try {
                this.pairInvite = (await this.apiFetch('/api/cluster/pair/invite'))?.invite || null;
            } catch (_) {
                this.pairInvite = null;
            }
        },

        async acceptInvite() {
            const device = this.discoveredDevices().find(
                (item) => item.node_id === this.pairInvite?.node_id,
            );
            this.pairInvite = null;
            if (device) await this.beginJoinAsJoiner(device);
        },

        async dismissInvite() {
            this.pairInvite = null;
            try {
                await this.apiFetch('/api/cluster/pair/invite', { method: 'DELETE' });
            } catch (_) {
                /* it expires on its own */
            }
        },

        cancelPairing() {
            this.pairing = { target: null, code: '', busy: false, error: '' };
        },

        async submitPairApproval(device) {
            const code = (this.pairing.code || '').trim();
            if (!device || this.pairing.busy) return;
            if (!/^\d{6}$/.test(code)) {
                this.pairing.error = window.t('cluster.pair.code_hint');
                return;
            }
            if (this.membershipPanelOpen && !this.membershipPairingReady()) {
                this.pairing.error = window.t('cluster.membership.waiting_request');
                return;
            }
            this.pairing.busy = true;
            this.pairing.error = '';
            try {
                await this.apiFetch(CLUSTER_API.pairApprove, {
                    method: 'POST',
                    body: JSON.stringify({ node_id: device.node_id, code }),
                });
                this.notify(
                    'success',
                    window.t('cluster.toast.peer_joined').replace('{name}', this.deviceName(device)),
                );
                this.cancelPairing();
                if (this.deviceHelperOnly(device)) {
                    // A helper daemon has one purpose, so there is no choice.
                    this.startAccelerator(device);
                } else if (this.membershipPanelOpen) {
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
                    this.choosePurpose(true);
                }
            } catch (error) {
                if (error?.status === 404 || error?.status === 409) {
                    this.pairing.error = window.t(
                        this.deviceHelperOnly(device)
                            ? 'cluster.pair.no_request_helper'
                            : 'cluster.pair.no_request',
                    );
                } else if (error?.status === 403 || error?.status === 429) {
                    this.pairing.error =
                        error.message ||
                        window.t('cluster.pair.lockout');
                } else {
                    this.pairing.error =
                        error?.message || window.t('cluster.err.pairing_failed');
                }
            } finally {
                this.pairing.busy = false;
            }
        },

        async submitPairDenial(device) {
            if (!device) return;
            try {
                await this.apiFetch(CLUSTER_API.pairDeny, {
                    method: 'POST',
                    body: JSON.stringify({ node_id: device.node_id }),
                });
                this.notify(
                    'info',
                    window.t('cluster.toast.join_request_denied').replace('{name}', this.deviceName(device)),
                );
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || window.t('cluster.err.deny_join'),
                );
            }
            if (this.pairing.target?.node_id === device?.node_id) {
                this.cancelPairing();
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
                window.t('cluster.join.other_mac')
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
            const host = addr.ip.includes(':') && !addr.ip.startsWith('[')
                ? `[${addr.ip}]` : addr.ip;
            return `${host}:${device.http_port || 8000}`;
        },

        async beginJoinAsJoiner(device) {
            const target = this.coordinatorAddrFor(device);
            if (!target) {
                this.notify(
                    'error',
                    window.t('cluster.toast.no_address').replace('{name}', this.deviceName(device)),
                );
                return;
            }
            await this.beginJoinAddr(target, this.deviceName(device));
        },

        async beginJoinAddr(coordinatorAddr, targetName) {
            if (this.join.busy) return;
            const revision = ++this.joinRevision;
            this.join.busy = true;
            try {
                const snapshot = await this.apiFetch(CLUSTER_API.pairJoin, {
                    method: 'POST',
                    body: JSON.stringify({ coordinator_addr: coordinatorAddr }),
                });
                if (revision !== this.joinRevision) return;
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
                if (revision !== this.joinRevision) return;
                this.join.busy = false;
                this.notify(
                    'error',
                    error?.message || window.t('cluster.err.reach_mac'),
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
            const revision = ++this.joinRevision;
            this.join.busy = true;
            try {
                const snapshot = await this.apiFetch(
                    CLUSTER_API.pairJoinCancel,
                    { method: 'POST' },
                );
                if (revision !== this.joinRevision) return;
                this.join = {
                    ...this.join,
                    ...(snapshot || { state: 'idle' }),
                    busy: false,
                    target_name: '',
                };
                if (!options.silent) {
                    this.notify(
                        snapshot?.cleanup_pending ? 'warning' : 'info',
                        snapshot?.cleanup_pending
                            ? window.t('cluster.toast.join_cancelled_pending')
                            : window.t('cluster.toast.join_cancelled'),
                    );
                }
            } catch (error) {
                if (revision !== this.joinRevision) return;
                this.join.busy = false;
                if (!options.silent) {
                    this.notify(
                        'error',
                        error?.message || window.t('cluster.err.cancel_join'),
                    );
                }
            }
        },

        async forgetJoinCleanup() {
            if (this.join.busy || !this.join.cleanup_pending) return;
            if (!this.confirmForgetCleanup) {
                this.confirmForgetCleanup = true;
                return;
            }
            this.confirmForgetCleanup = false;
            const revision = ++this.joinRevision;
            this.join.busy = true;
            try {
                const snapshot = await this.apiFetch(CLUSTER_API.pairJoinCleanup, {
                    method: 'DELETE',
                });
                if (revision !== this.joinRevision) return;
                this.join = { ...this.join, ...snapshot, busy: false };
                this.notify('info', window.t('cluster.join.cleanup_forgotten'));
            } catch (error) {
                if (revision !== this.joinRevision) return;
                this.join.busy = false;
                this.notify('error', error?.message || window.t('cluster.join.forget_cleanup_error'));
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
                this.manualError = window.t('cluster.manual.invalid_addr');
                return;
            }
            const ip = match[1];
            const port = match[2] ? parseInt(match[2], 10) : 8000;
            const octetsOk = ip.split('.').every((part) => Number(part) <= 255);
            if (!octetsOk || !(port >= 1 && port <= 65535)) {
                this.manualError = window.t('cluster.manual.out_of_range');
                return;
            }
            this.manualBusy = true;
            this.manualError = '';
            try {
                const result = await this.apiFetch(CLUSTER_API.manualDevice, {
                    method: 'POST',
                    body: JSON.stringify({ ip, port }),
                });
                await this.refreshDevices();
                if (result && result.verified) {
                    const name = result.peer?.friendly_name || ip;
                    this.notify('success', window.t('cluster.toast.found_peer').replace('{name}', name).replace('{ip}', ip));
                    if (options.beginJoin !== false) {
                        // First-cluster setup may join in either direction.
                        // An existing coordinator only discovers here: the
                        // new Mac initiates and this cluster approves its code.
                        await this.beginJoinAddr(`${ip}:${port}`, name);
                    }
                } else {
                    this.notify(
                        'warning',
                        window.t('cluster.toast.no_node_answered'),
                    );
                }
                this.manualAddr = '';
            } catch (error) {
                this.manualError =
                    error?.message || window.t('cluster.err.add_address');
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

        // A paired Mac either splits models with this one (checks, plan) or
        // runs the first layers of this Mac's prefills (Prefill Accelerator).
        choosePurpose(auto = false) {
            this.stage = 'purpose';
            this.purposeAuto = auto;
        },

        async refreshHelperRole() {
            try {
                this.helperRole = await this.apiFetch('/admin/api/prefill-helper/lending');
            } catch (_) {
                this.helperRole = null;
            }
        },

        helperRoleHost() {
            const role = this.helperRole;
            return (role?.active ? role.host : '') || role?.helps?.name || '';
        },

        // Forgetting this Mac's host ends the helper role; the host can no
        // longer reach it.
        helperHostDevice() {
            const nodeId = this.helperRole?.helps?.node_id;
            return this.pairedDevices().find((item) => item.node_id === nodeId) || null;
        },

        helperRoleText() {
            const role = this.helperRole || {};
            const key = role.active ? 'cluster.helper.active' : 'cluster.helper.idle';
            return window.t(key)
                .replaceAll('{host}', this.helperRoleHost())
                .replace('{n}', role.layers ?? '')
                .replace('{model}', String(role.model || '').split('/').pop());
        },

        async stopHelping() {
            try {
                await this.apiFetch('/admin/api/prefill-helper/lending/stop', { method: 'POST' });
            } catch (_) {
                /* refreshed below */
            }
            await this.refreshHelperRole();
        },

        // A paired Mac that a model already uses as its Prefill Accelerator
        // opens on that screen; Change setup goes back to the choice.
        async restoreAccelerator() {
            if (
                this.acceleratorRestored ||
                !this.devicesLoaded ||
                !this.deploymentsLoaded ||
                !this.pairedDevices().length
            ) {
                return;
            }
            this.acceleratorRestored = true;
            // A model here using that Mac makes this Mac its host, even when an
            // old record says this Mac helps it.
            const open = () => ['device_card', 'helper'].includes(this.wizardState());
            if (!open()) return;
            let payload;
            try {
                payload = await this.apiFetch('/admin/api/models');
            } catch (_) {
                return;
            }
            const used = new Set(
                (payload?.models || []).map((model) => model.settings?.prefill_helper_node_id),
            );
            // A helper daemon is only ever a Prefill Accelerator; one that
            // answers comes first.
            const helpers = this.pairedDevices().filter((item) => this.deviceHelperOnly(item));
            const device =
                this.pairedDevices().find((item) => used.has(item.node_id)) ||
                helpers.find((item) => item.state !== 'dead') ||
                helpers[0];
            if (device && open()) this.startAccelerator(device, { asHost: true });
        },

        async startAccelerator(device, { asHost = false } = {}) {
            if (!asHost && device && this.helperRole?.helps?.node_id === device.node_id) {
                // This Mac is that Mac's Prefill Accelerator, not its host.
                this.stage = null;
                return;
            }
            this.stage = 'accelerator';
            this.accelerator = {
                nodeId: device.node_id,
                running: true,
                result: null,
                error: '',
                copies: null,
                deleting: [],
            };
            this.loadAcceleratorCopies();
            try {
                const result = await this.apiFetch(
                    '/admin/api/prefill-helper/check?node_id=' +
                        encodeURIComponent(device.node_id),
                );
                if (this.accelerator.nodeId === device.node_id) {
                    this.accelerator.result = result;
                }
            } catch (error) {
                if (this.accelerator.nodeId === device.node_id) {
                    this.accelerator.error =
                        error?.message || window.t('cluster.err.probe_failed');
                }
            } finally {
                if (this.accelerator.nodeId === device.node_id) {
                    this.accelerator.running = false;
                }
            }
        },

        // Model copies this Mac made on the helper; its own model folders never show here.
        async loadAcceleratorCopies(remove = '') {
            const nodeId = this.accelerator.nodeId;
            const query = 'node_id=' + encodeURIComponent(nodeId)
                + (remove ? '&name=' + encodeURIComponent(remove) : '');
            if (remove) this.accelerator.deleting = [...this.accelerator.deleting, remove];
            try {
                const result = await this.apiFetch(
                    '/admin/api/prefill-helper/copies' + (remove ? '/delete?' : '?') + query,
                    remove ? { method: 'POST' } : {},
                );
                if (this.accelerator.nodeId === nodeId) this.accelerator.copies = result;
            } catch (_) {
                // A failed delete keeps the list as it was.
                if (!remove && this.accelerator.nodeId === nodeId) this.accelerator.copies = null;
            } finally {
                if (remove && this.accelerator.nodeId === nodeId) {
                    this.accelerator.deleting = this.accelerator.deleting.filter((name) => name !== remove);
                }
            }
        },

        acceleratorName() {
            const device = this.pairedDevices().find(
                (item) => item.node_id === this.accelerator.nodeId,
            );
            return this.accelerator.result?.name || this.deviceName(device);
        },

        acceleratorRows() {
            const rows = this.accelerator.result?.rows || [];
            const name = this.acceleratorName();
            const t = (key) => window.t('cluster.accelerator.' + key);
            const size = (bytes) => this.formatSizeBytes(Number(bytes || 0));
            // A helper daemon answers on its control channel instead of SSH.
            const helper = rows.some((row) => row.key === 'control');
            return rows.map((row) => {
                let detail = row.detail || '';
                let label = t('row_' + row.key);
                if (['ssh', 'control'].includes(row.key) && row.status === 'pass') {
                    detail = t(row.key + '_ok').replace('{target}', row.detail);
                } else if (row.key === 'link') {
                    const mbps = Number(row.speed_mbps || 0);
                    const speed = mbps >= 1000 ? `${mbps / 1000} Gb` : `${mbps} Mb`;
                    const ethernet = t('link_ethernet').replace('{speed}', speed);
                    if (['thunderbolt', 'rdma'].includes(row.detail)) {
                        detail = t('link_fast');
                    } else if (row.status === 'pass') {
                        detail = ethernet;
                    } else if (mbps >= 2500) {
                        detail = t('link_ethernet_ok').replace('{speed}', speed);
                    } else {
                        const kind = row.detail === 'wifi' ? 'Wi-Fi' : (mbps ? ethernet : 'Ethernet');
                        detail = t(helper ? 'link_slow_helper' : 'link_slow').replace('{kind}', kind);
                    }
                } else if (row.key === 'version' && helper) {
                    // The helper has its own version; an error text stays as is.
                    label = t('row_version_helper');
                    if (row.families) {
                        detail = t(row.status === 'fail' ? 'version_mismatch_helper' : 'version_ok_helper')
                            .replace('{version}', row.detail || '?');
                    }
                } else if (row.key === 'version') {
                    detail = row.status === 'pass'
                        ? t('version_ok').replace('{version}', row.detail)
                        : t('version_mismatch')
                            .replaceAll('{name}', name)
                            .replace('{version}', row.detail || '?')
                            .replace('{expected}', row.expected || '?');
                } else if (row.key === 'memory' && row.gpus) {
                    detail = t('memory_detail_helper')
                        .replace('{size}', size(row.bytes))
                        .replace('{n}', row.gpus);
                } else if (row.key === 'memory') {
                    detail = t('memory_detail').replace('{size}', size(row.bytes));
                } else if (row.key === 'disk') {
                    detail = t('disk_detail').replace('{size}', size(row.bytes));
                } else if (row.key === 'power') {
                    detail = row.status === 'pass' ? t('power_ok') : t('power_battery');
                }
                return {
                    key: row.key,
                    status: row.status,
                    label,
                    detail,
                    fix: row.fix || '',
                };
            });
        },

        acceleratorReady() {
            const rows = this.accelerator.result?.rows || [];
            return rows.length > 0 && rows.every((row) => row.status !== 'fail');
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
        },

        async saveSSHUser(device) {
            const nodeId = device.node_id;
            if (this.sshUserSaving[nodeId]) return;
            const value = String(this.sshUserDrafts[nodeId] ?? device.ssh_user ?? '').trim();
            this.sshUserSaving = {...this.sshUserSaving, [nodeId]: true};
            try {
                const saved = await this.apiFetch(CLUSTER_API.sshUser(nodeId), {
                    method: 'PUT',
                    body: JSON.stringify({ssh_user: value || null}),
                });
                device.ssh_user = saved.ssh_user;
                this.sshUserDrafts = {...this.sshUserDrafts, [nodeId]: value};
                // A plan and its probes are tied to the previous SSH identity.
                ++this.planRequestRevision;
                this.plan = null;
                this.planProposal = null;
                this.checks.probes = {};
                this.checks.benchmark = null;
                this.checks.started = false;
                await this.refreshDevices();
                this.notify('success', window.t('cluster.device.ssh_user_saved'));
            } catch (error) {
                this.notify('error', error?.message || window.t('cluster.device.ssh_user_error'));
            } finally {
                this.sshUserSaving = {...this.sshUserSaving, [nodeId]: false};
            }
        },

        sshTargetFor(device) {
            const enrolled = String(device?.ssh_target || '');
            const separator = enrolled.lastIndexOf('@');
            const user = device?.ssh_user || (separator > 0 ? enrolled.slice(0, separator) : '');
            const withUser = (target) => user
                ? `${user}@${String(target).replace(/^[^@]+@/, '')}`
                : String(target);
            const addrs = Array.isArray(device?.addrs) ? device.addrs : [];
            // A bare fe80:: link-local address has no scope id here, so SSH
            // to it has no route — prefer any routable address first.
            const usable = addrs.filter(
                (addr) => addr && addr.ip && !String(addr.ip).startsWith('fe80::'),
            );
            // Pairing pins every address. Select a verified address before
            // falling back to the enrolled target, keeping the same login.
            const verified = usable.find(
                // The shared SSH policy forces AddressFamily=inet.
                (addr) => !String(addr.ip).includes(':')
                    && device?.address_health?.[addr.ip]?.state === 'verified',
            );
            if (verified) return withUser(verified.ip);
            if (device?.ssh_target) return withUser(device.ssh_target);
            const first = usable[0] || addrs.find((addr) => addr && addr.ip);
            return withUser(first ? first.ip : this.deviceName(device));
        },

        async probePeer(peer) {
            const ssh = this.sshTargetFor(peer);
            try {
                const result = await this.apiFetch(CLUSTER_API.peerProbe, {
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
                        error: error?.message || window.t('cluster.err.probe_failed'),
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
                        window.t('cluster.checks.bench_need_model'),
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
                        error: this.planError || window.t('cluster.err.calibration_failed'),
                    };
                }
            } catch (error) {
                this.checks.benchmark = {
                    ok: false,
                    error: error?.message || window.t('cluster.err.benchmark_failed'),
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
                const warnings = probed.flatMap((probe) =>
                    probe?.ok && Array.isArray(probe.result?.runtime_warnings)
                        ? probe.result.runtime_warnings
                        : [],
                );
                rows.push({
                    key: 'ssh',
                    label: window.t('cluster.checks.ssh_label'),
                    status: !this.checks.started
                        ? 'pending'
                        : running
                        ? 'running'
                        : peers.length && allPass(probed.map((p) => p?.ok))
                        ? warnings.length
                            ? 'warn'
                            : 'pass'
                        : failures.length
                        ? 'fail'
                        : 'running',
                    detail: failures.length
                        ? window.t('cluster.checks.ssh_fail')
                              .replace('{names}', failures.map((peer) => this.deviceName(peer)).join(', '))
                        : warnings.length
                        ? warnings.join(' ')
                        : window.t('cluster.checks.ssh_ok'),
                    fix: failures.length
                        ? window.t('cluster.checks.ssh_fix')
                        : warnings.length
                        ? window.t('cluster.checks.warn_fix')
                        : '',
                });
            }

            // 2. Model presence (resolved once a model is chosen in the plan
            //    step; the wizard annotates presence per device there).
            {
                const model = this.selectedModel();
                let status = 'skipped';
                let detail = window.t('cluster.checks.model_skipped');
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
                        detail = window.t('cluster.checks.model_missing')
                        .replace('{model}', this.shortModelName(model))
                        .replace('{names}', missing.map((device) => this.deviceName(device)).join(', '));
                    } else {
                        status = 'pass';
                        detail = window.t('cluster.checks.model_on_every').replace('{model}', this.shortModelName(model));
                    }
                }
                rows.push({
                    key: 'model',
                    label: window.t('cluster.checks.model_label'),
                    status,
                    detail,
                    fix: window.t('cluster.checks.model_fix'),
                });
            }

            // 3. Version parity (from the discovery snapshot, live).
            {
                const mismatches = this.versionMismatches();
                rows.push({
                    key: 'version',
                    label: window.t('cluster.checks.version_label'),
                    status: mismatches.length ? 'fail' : 'pass',
                    detail: mismatches.length
                        ? mismatches
                              .map(
                                  (m) =>
                                      window.t('cluster.checks.version_mismatch').replace('{name}', m.name).replace('{peer}', String(m.peerVersion)).replace('{self}', String(m.selfVersion)),
                              )
                              .join(' ')
                        : window.t('cluster.checks.version_ok'),
                    fix: window.t('cluster.checks.version_fix'),
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
                        label: window.t('cluster.checks.rdma_label'),
                        status: 'skipped',
                        detail:
                            window.t('cluster.checks.rdma_skipped'),
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
                        label: window.t('cluster.checks.rdma_label'),
                        status: this.checks.running
                            ? 'running'
                            : failing.length
                            ? 'fail'
                            : 'pass',
                        detail: failing.length
                            ? window.t('cluster.checks.rdma_fail')
                                  .replace('{names}', failing.map((peer) => this.deviceName(peer)).join(', '))
                            : window.t('cluster.checks.rdma_ok'),
                        fix: window.t('cluster.checks.rdma_fix'),
                    });
                }
            }

            // 5. Benchmark (explicit — it loads the chips for a few seconds).
            {
                const bench = this.checks.benchmark;
                rows.push({
                    key: 'benchmark',
                    label: window.t('cluster.checks.bench_label'),
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
                            ? window.t('cluster.checks.bench_ok')
                            : bench.error
                        : !this.selectedModelPath
                        ? window.t('cluster.checks.bench_runs_auto')
                        : window.t('cluster.checks.bench_measuring'),
                    fix: window.t('cluster.checks.bench_fix'),
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
                let detail = window.t('cluster.checks.beacon_checking');
                let fix =
                    window.t('cluster.checks.beacon_fix');
                if (health) {
                    if (health.multicast_rx_within_5s) {
                        status = 'pass';
                        detail = window.t('cluster.checks.beacon_ok');
                    } else {
                        status = 'warn';
                        detail =
                            window.t('cluster.checks.beacon_none');
                    }
                } else if (this.discoveryHealthUnsupported) {
                    status = 'skipped';
                    detail =
                        window.t('cluster.checks.beacon_unsupported');
                    fix = '';
                } else if (this.checks.running) {
                    status = 'running';
                }
                rows.push({
                    key: 'multicast',
                    label: t('cluster.checks.beacon_label'),
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
                ['pass', 'warn'].includes(byKey.ssh?.status) &&
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
                const payload = await this.apiFetch(CLUSTER_API.nodeRoles);
                this.roleOptions = payload?.roles || [];
            } catch (error) {
                // The mirrored fallback constants keep the budget labels
                // rendering; planning itself re-derives roles server-side.
            }
        },

        roleSpecFor(key) {
            return (
                this.roleOptions.find((role) => role.key === key) ||
                CLUSTER_ROLE_FALLBACK[key] ||
                CLUSTER_ROLE_FALLBACK.headless
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
            const measured = this.measuredNodeBudgets[device.node_id];
            const capacity = Number(measured?.capacity_bytes) || (this.deviceRamGb(device) || 0) * 1024 ** 3;
            if (!capacity) return '';
            const role = this.nodeRole(device.node_id, !!device.is_self);
            const usable = Math.max(
                0,
                capacity - this.reserveBytesFor(role, capacity),
            );
            return window.t('cluster.role.usable').replace('{gb}', String(Math.round(usable / 1024 ** 3))).replace('{role}', this.roleLabel(role));
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
            for (const peer of this.splitDevices()) {
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
                const payload = await this.apiFetch(CLUSTER_API.models, {
                    method: 'POST',
                    body: JSON.stringify({ hosts: this.inventoryHosts() }),
                });
                this.modelOptions = payload?.models || [];
            } catch (error) {
                this.modelsError =
                    error?.message || window.t('cluster.err.list_models');
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
            let name = segments.pop() || raw || window.t('cluster.model.this_model');
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
            if (have >= total) return t('cluster.models.on_every_mac');
            return t('cluster.models.partial')
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
                    label: window.t('cluster.exec.interactive'),
                    limits: window.t('cluster.exec.interactive_limits'),
                    detail: window.t('cluster.exec.interactive_detail'),
                },
                {
                    key: 'balanced',
                    label: window.t('cluster.exec.balanced'),
                    limits: window.t('cluster.exec.balanced_limits'),
                    detail: window.t('cluster.exec.balanced_detail'),
                },
                {
                    key: 'throughput',
                    label: window.t('cluster.exec.throughput'),
                    limits: window.t('cluster.exec.throughput_limits'),
                    detail: window.t('cluster.exec.throughput_detail'),
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
            return [
                {
                    key: 'auto',
                    label: t('cluster.strategy.auto'),
                    disabled: false,
                    disabledReason: '',
                },
                {
                    key: 'tensor',
                    label: t('cluster.strategy.tensor'),
                    disabled: tensorDisabled,
                    disabledReason: !tensorDisabled
                        ? ''
                        : nodeCount < 2
                        ? t('cluster.strategy.tensor_needs_two')
                        : t('cluster.strategy.tensor_unsupported'),
                },
                {
                    key: 'pipeline',
                    label: t('cluster.strategy.pipeline'),
                    disabled: pipelineUnsupported,
                    disabledReason: pipelineUnsupported
                        ? t('cluster.strategy.pipeline_unsupported')
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
            return t(`cluster.strategy.hint.${this.planStrategy}`);
        },

        setPlanStrategy(key) {
            const option = this.strategyOptions().find(
                (item) => item.key === key,
            );
            if (!option || option.disabled || this.planLoading) return;
            if (this.planStrategy === key) return;
            this.planStrategy = key;
            if (this.selectedModelPath) this.runPlan();
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
                const payload = await this.apiFetch(CLUSTER_API.catalogue, {
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

        // Tensor plans give every node ALL layers with a tensor_parallel_rank;
        // the contiguous-range split bar would lie about them.
        planIsTensor() {
            return (this.plan?.tensor_parallel_size || 1) > 1;
        },

        tensorShareLabel() {
            return t('cluster.split.tensor_share').replace(
                '{count}',
                String(this.plan?.tensor_parallel_size || 1),
            );
        },

        tensorCaptionLabel() {
            return t('cluster.split.tensor_caption').replace(
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
                });
            }
            for (const peer of this.splitDevices()) {
                nodes.push({
                    node_id: peer.node_id,
                    capacity_bytes:
                        (this.deviceRamGb(peer) || 0) * 1024 ** 3,
                    role: this.nodeRole(peer.node_id, false),
                    memory_guard_tier: 'balanced',
                });
            }
            return nodes.map((node) => {
                const measured = this.measuredNodeBudgets[node.node_id];
                return measured && measured.role === node.role
                    ? {...node, capacity_bytes: measured.capacity_bytes, reserve_bytes: measured.reserve_bytes}
                    : node;
            }).filter((node) => node.capacity_bytes > 0);
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
                const hosts = this.deploymentHosts();
                const measured = await this.measurePlanNodes(hosts, this.planNodes());
                if (revision !== this.planRequestRevision) return null;
                const nodes = measured.map((node) => ({
                    ...node,
                    ...(priorPerformance.has(node.node_id)
                        ? { performance: priorPerformance.get(node.node_id) }
                        : {}),
                }));
                const body = {
                    model_path: this.selectedModelPath,
                    nodes,
                    hosts,
                    path_map: Object.fromEntries((model?.locations || [])
                        .filter((loc) => loc.node_id && loc.model_path)
                        .map((loc) => [loc.node_id, loc.model_path])),
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
                    CLUSTER_API.autoconfigure,
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
                        window.t('cluster.err.plan_missing_signature'),
                    );
                }
                if (
                    proposal.activation.approved_placement !==
                    proposal.plan.placement_signature
                ) {
                    throw new Error(
                        window.t('cluster.err.plan_signature_mismatch'),
                    );
                }
                this.planProposal = proposal;
                this.plan = proposal.plan;
                const probe = proposal.performance_probe;
                this.checks.benchmark =
                    probe?.ok === true
                        ? { ok: true, result: probe }
                        : {
                              ok: false,
                              result: probe || null,
                              error:
                                  probe?.reason ||
                                  window.t('cluster.err.calibration_incomplete'),
                          };
                return proposal;
            } catch (error) {
                if (revision !== this.planRequestRevision) return null;
                this.planError =
                    error?.message || window.t('cluster.err.build_layer_split');
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

        // Server-written, already redacted. They name what to fix before
        // activating, like a Thunderbolt cable the cluster cannot use.
        planWarnings() {
            const warnings = this.planProposal?.warnings;
            return Array.isArray(warnings) ? warnings : [];
        },

        planTotalLayers() {
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
                : assignment.node_id || window.t('cluster.plan.rank').replace('{n}', String(assignment.rank));
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
                ? window.t('cluster.backend.jaccl')
                : window.t('cluster.backend.ring');
        },

        deploymentFabricLabel() {
            const backend = String(
                this.configuredDeployment()?.backend || this.resolvedBackend(),
            );
            return backend.startsWith('jaccl')
                ? window.t('cluster.deploy.fabric_jaccl')
                : window.t('cluster.deploy.fabric_ring');
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

        proposalCanProceed(proposal) {
            return proposal?.ready_to_activate === true || proposal?.ready_to_stage === true;
        },

        async measurePlanNodes(hosts, nodes) {
            const roles = Object.fromEntries(nodes.map((node) => [node.node_id, node.role]));
            const result = await this.apiFetch(CLUSTER_API.nodeBudgets, {
                method: 'POST',
                body: JSON.stringify({hosts: hosts.map(({node_id, ssh, python_executable}) => ({node_id, ssh, python_executable})), roles}),
            });
            const measured = result?.nodes;
            if (!Array.isArray(measured) || measured.length !== nodes.length ||
                new Set(measured.map((node) => node.node_id)).size !== nodes.length) {
                throw new Error(window.t('cluster.err.memory_probe_incomplete'));
            }
            const resolved = nodes.map((node) => {
                const budget = measured.find((item) => item.node_id === node.node_id);
                if (!budget || budget.unusable || !(Number(budget.capacity_bytes) > 0)) {
                    throw new Error(window.t('cluster.err.memory_budget_unavailable').replace('{node}', node.node_id));
                }
                return {...node, capacity_bytes: Number(budget.capacity_bytes), reserve_bytes: Number(budget.reserve_bytes || 0)};
            });
            this.measuredNodeBudgets = Object.fromEntries(resolved.map((node) => [node.node_id, node]));
            return resolved;
        },

        async activatePlan() {
            const activation = this.activationRequestBody();
            if (!this.plan || !activation || this.activateBusy || !this.proposalCanProceed(this.planProposal)) return;
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

        async stageModelToPeers(activation, purpose = 'plan') {
            this.stagingPurpose = purpose;
            this.stopStagingPoll();
            const generation = ++this.stagingGeneration;
            this.stagingClaimed = false;
            this.stagingJob = null;
            this.stagingError = '';
            this.stagingActivation = activation;
            let job;
            try {
                job = await this.apiFetch(CLUSTER_API.stage, {
                    method: 'POST',
                    body: JSON.stringify({
                        activation,
                        parallel: 4,
                    }),
                });
            } catch (error) {
                if (generation !== this.stagingGeneration) return;
                this.stopStagingPoll();
                if (error?.status === 409) {
                    // Signature drift between plan and stage — identical
                    // handling to activation's 409 below.
                    this.notify(
                        'warning',
                        window.t('cluster.toast.plan_changed'),
                    );
                    this.activateBusy = false;
                    this.stagingActivation = null;
                    await this.refreshActivationProposal(this.stagingPurpose);
                } else if (error?.status === 404) {
                    // A server older than /stage: degrade to the pre-staging
                    // behavior instead of bricking activation — the
                    // activation preflight reports whatever is still missing.
                    this.notify('warning', t('cluster.staging.unsupported'));
                    await this.postActivation(activation, this.stagingPurpose);
                } else {
                    this.stagingActivation = null;
                    this.failStaging(
                        error?.message || t('cluster.staging.failed'),
                    );
                    this.activateBusy = false;
                }
                return;
            }
            if (generation !== this.stagingGeneration) return;
            this.stagingJob = job;
            this.stagingTimer = setInterval(
                () => this.pollStagingJob(),
                CLUSTER_POLL_MS,
            );
        },

        async pollStagingJob() {
            if (this.stagingPollBusy || this.stagingClaimed) return;
            const generation = this.stagingGeneration;
            const jobId = this.stagingJob?.job_id;
            if (!jobId) {
                this.stopStagingPoll();
                return;
            }
            let snapshot;
            this.stagingPollBusy = true;
            try {
                snapshot = await this.apiFetch(
                    CLUSTER_API.stageJob(jobId),
                );
            } catch (error) {
                if (generation !== this.stagingGeneration) return;
                // 404 = the coordinator restarted and the in-memory job is
                // gone. Re-POSTing /stage is safe (verified files skip), so
                // pressing Activate again is the resume.
                this.stopStagingPoll();
                this.failStaging(
                    error?.status === 404
                        ? t('cluster.staging.lost')
                        : error?.message || t('cluster.staging.failed'),
                );
                this.activateBusy = false;
                return;
            } finally {
                this.stagingPollBusy = false;
            }
            if (generation !== this.stagingGeneration || jobId !== this.stagingJob?.job_id || this.stagingClaimed) return;
            this.stagingJob = snapshot;
            if (snapshot.status === 'completed') {
                if (snapshot.ready !== true) {
                    this.stopStagingPoll();
                    this.failStaging(window.t('cluster.err.staging_not_ready'));
                    this.activateBusy = false;
                    return;
                }
                this.stagingClaimed = true;
                this.stopStagingPoll();
                const activation = this.stagingActivation;
                this.stagingActivation = null;
                await this.postActivation(activation, this.stagingPurpose); // phase 2, exactly once
            } else if (snapshot.status === 'failed') {
                // Other nodes may still have completed; per-node errors stay
                // visible in stagingJob.nodes next to the banner. Pressing
                // Activate retries — only missing files move again.
                this.stopStagingPoll();
                this.failStaging(
                    snapshot.error || t('cluster.staging.failed'),
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
            this.stagingError = message || t('cluster.staging.failed');
            this.notify('error', this.stagingError);
        },

        // Client-side dismiss only — there is no cancel endpoint. The copy
        // finishes server-side regardless, harmlessly: the next attempt
        // size-verifies and skips whatever already landed.
        dismissStaging() {
            this.stagingGeneration += 1;
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
                    ? t('cluster.staging.ready')
                    : t('cluster.staging.already_there');
            }
            if (status === 'failed') {
                return node.error || t('cluster.staging.failed');
            }
            if (status === 'copying') {
                if (!(node.files_total > 0)) {
                    return t('cluster.staging.checking');
                }
                return t('cluster.staging.copying')
                    .replace('{done}', String(node.files_completed || 0))
                    .replace('{total}', String(node.files_total || 0));
            }
            return t('cluster.staging.waiting');
        },

        stagingOverallLabel() {
            const nodes = this.stagingNodes();
            const done = nodes.filter(
                (node) => node.status === 'ready',
            ).length;
            return t('cluster.staging.title')
                .replace('{done}', String(done))
                .replace('{total}', String(nodes.length));
        },

        stagingButtonLabel() {
            const missing = this.nodesMissingModel().length;
            return t(
                missing === 1
                    ? 'cluster.staging.button_copying_one'
                    : 'cluster.staging.button_copying',
            ).replace('{count}', String(missing));
        },

        // Phase 2 — the pre-staging activation path, unchanged.
        async refreshActivationProposal(purpose) {
            if (purpose === 'membership') {
                this.membershipProposal = null;
                await this.previewMembershipExpansion();
            } else {
                await this.runPlan();
            }
        },

        async postActivation(activation = this.activationRequestBody(), purpose = 'plan') {
            const generation = this.stagingGeneration;
            if (!activation) {
                this.activateBusy = false;
                this.notify('error', window.t('cluster.err.activation_missing'));
                return;
            }
            try {
                await this.apiFetch(CLUSTER_API.deployments, {
                    method: 'POST',
                    body: JSON.stringify(activation),
                });
                if (generation !== this.stagingGeneration) return;
                this.notify(
                    'success',
                    window.t('cluster.toast.activated'),
                );
                this.stage = null;
                this.choosingModel = false;
                this.selectedDeploymentId = activation.deployment_id || '';
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
                if (generation !== this.stagingGeneration) return;
                if (error?.status === 409) {
                    this.notify(
                        'warning',
                        window.t('cluster.toast.plan_changed'),
                    );
                    this.stagingActivation = null;
                    await this.refreshActivationProposal(purpose);
                } else {
                    this.notify(
                        'error',
                        error?.message || window.t('cluster.err.activation_failed'),
                    );
                }
            } finally {
                if (generation === this.stagingGeneration) {
                    this.stopStagingPoll();
                    this.activateBusy = false;
                }
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
            if (Number(execution.max_kv_size) > 0) {
                body.max_kv_size = Number(execution.max_kv_size);
            }
            if (Number(execution.ring_connections_per_ip) > 0) {
                body.ring_connections_per_ip = Number(
                    execution.ring_connections_per_ip,
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
                const preview = await this.apiFetch(CLUSTER_API.replan, {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
                const signature = preview?.plan?.placement_signature;
                if (typeof signature !== 'string' || signature.length < 16) {
                    throw new Error(window.t('cluster.err.replan_unsigned'));
                }
                this.executionReplan = { profile, body, preview };
            } catch (error) {
                this.executionReplan = null;
                this.notify(
                    'error',
                    error?.message || window.t('cluster.err.preview_profile'),
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
                await this.apiFetch(CLUSTER_API.replan, {
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
                    window.t('cluster.toast.profile_applied'),
                );
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message ||
                        window.t('cluster.err.apply_replan'),
                );
            } finally {
                this.executionReplanBusy = false;
            }
        },

        // =================================================================
        // Active membership — pair first, then sign one N-node re-plan.
        // =================================================================
        beginMembershipPairing(device) {
            if (!device?.node_id || this.pairing.busy) return;
            this.membershipPanelOpen = true;
            this.beginPairing(device);
        },

        membershipPairingReady() {
            return this.pendingApprovals().some(
                (device) => device.node_id === this.pairing.target?.node_id,
            );
        },

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
                    CLUSTER_API.nodeBudgets,
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
                            .map((node) =>
                                window.t('cluster.err.node_probe')
                                    .replace('{node}', node.node_id)
                                    .replace(
                                        '{message}',
                                        node.error
                                            || window.t('cluster.err.memory_probe_unavailable'),
                                    ),
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
                        ...(performance ? { performance } : {}),
                    };
                });
                const execution = this.deploymentExecution(deployment) || {};
                const proposal = await this.apiFetch(
                    CLUSTER_API.autoconfigure,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            deployment_id: deployment.deployment_id,
                            path_map: deployment.path_map || {},
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
                        }),
                    },
                );
                this.membershipProposal = proposal;
                if (!this.proposalCanProceed(proposal)) {
                    this.membershipError =
                        proposal?.fabric_blocker ||
                        proposal?.preflight ||
                        window.t('cluster.membership.not_ready');
                }
            } catch (error) {
                this.membershipError =
                    error?.message || window.t('cluster.err.preview_expanded');
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
                : assignment?.node_id || window.t('cluster.plan.rank').replace('{n}', String(assignment?.rank ?? '?'));
        },

        membershipPlanDetail(assignment) {
            const gib = Number(assignment?.planned_weight_bytes || 0) / (1024 ** 3);
            const tp = Number(assignment?.tensor_parallel_size || 1);
            return tp > 1
                ? window.t('cluster.membership.tensor_rank').replace('{rank}', String(Number(assignment?.tensor_parallel_rank || 0) + 1)).replace('{tp}', String(tp)).replace('{gib}', gib.toFixed(1))
                : window.t('cluster.membership.layers').replace('{start}', String(assignment?.start_layer ?? 0)).replace('{end}', String(assignment?.end_layer ?? 0)).replace('{gib}', gib.toFixed(1));
        },

        async applyMembershipExpansion() {
            const proposal = this.membershipProposal;
            const activation = proposal?.activation;
            if (
                !this.proposalCanProceed(proposal) ||
                !activation ||
                this.membershipBusy ||
                this.activateBusy
            ) {
                return;
            }
            this.activateBusy = true;
            await this.stageModelToPeers(activation, 'membership');
        },

        hydratePlannerFromDeployment(deployment) {
            const execution = this.deploymentExecution(deployment) || {};
            this.executionProfile = String(
                execution.profile || deployment?.execution?.profile || 'balanced',
            );
            this.promptCacheSsd = Boolean(
                execution.prompt_cache_ssd ?? deployment?.prompt_cache_ssd ?? true,
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
                await this.apiFetch(CLUSTER_API.deploymentUnload(id), {
                    method: 'POST',
                });
                this.notify(
                    'success',
                    window.t('cluster.toast.weights_unloaded'),
                );
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message || window.t('cluster.err.unload_failed'),
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
                await this.apiFetch(CLUSTER_API.deploymentLoad(id), {
                    method: 'POST',
                });
                this.notify(
                    'success',
                    window.t('cluster.toast.weights_loaded'),
                );
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message || window.t('cluster.err.load_failed'),
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
                await this.apiFetch(CLUSTER_API.deployment(id), {
                    method: 'DELETE',
                });
                this.confirmChangeModelFor = '';
                this.resetModelPicker();
                this.choosingModel = true;
                await this.refreshDeployments();
                await this.refreshRuntime();
                this.enterPlan();
                this.notify(
                    'info',
                    window.t('cluster.toast.model_changed'),
                );
            } catch (error) {
                this.notify(
                    error?.status === 409 ? 'warning' : 'error',
                    error?.message || window.t('cluster.err.change_model'),
                );
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        async forgetCluster(device = null) {
            if (this.clusterLifecycleBusy) return;
            const nodeId = device?.node_id;
            const confirmation = nodeId ? `node:${nodeId}` : 'all';
            if (this.confirmForgetFor !== confirmation) {
                this.confirmForgetFor = confirmation;
                return;
            }
            this.confirmForgetFor = '';
            this.clusterLifecycleBusy = true;
            try {
                await this.apiFetch('/admin/api/cluster/forget' +
                    (nodeId ? `?node_id=${encodeURIComponent(nodeId)}` : ''), {method: 'DELETE'});
                this.notify('info', window.t('cluster.toast.forgotten_local'));
                await this.refreshDevices();
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify('error', error?.message || window.t('cluster.err.unpair'));
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        async deactivateDeployment(deployment, localOnly = false) {
            const id = deployment?.deployment_id;
            if (!id || this.clusterLifecycleBusy) return;
            const confirmation = localOnly ? `${id}:local` : id;
            if (this.confirmDeactivateFor !== confirmation) {
                this.confirmDeactivateFor = confirmation;
                return;
            }
            this.confirmDeactivateFor = '';
            this.confirmChangeModelFor = '';
            this.confirmUnloadFor = '';
            this.clusterLifecycleBusy = true;
            try {
                await this.apiFetch(CLUSTER_API.deployment(id) + (localOnly ? "?local_only=true" : ""), {
                    method: 'DELETE',
                });
                this.notify('info', window.t(localOnly ? 'cluster.toast.forgotten_local' : 'cluster.toast.deactivated'));
                await this.refreshDeployments();
                await this.refreshRuntime();
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || window.t('cluster.err.deactivate'),
                );
            } finally {
                this.clusterLifecycleBusy = false;
            }
        },

        // =====================================================================
        // Advanced tools — CUDA enrollment/fabric proof and diagnostics
        // =====================================================================
        async toggleAdvancedTools() {
            this.advancedOpen = !this.advancedOpen;
            if (this.advancedOpen) await this.loadCudaJoinStatus();
        },

        cudaNodes() {
            return Array.isArray(this.cudaJoin.status?.nodes)
                ? this.cudaJoin.status.nodes
                : [];
        },

        cudaJoinSuggestedIp() {
            const explicit = String(this.cudaJoin.controllerIp || '').trim();
            if (explicit) return explicit;
            const addresses = this.selfDevice()?.addrs || [];
            const local = addresses.find(
                (item) =>
                    item?.ip &&
                    /^\d{1,3}(?:\.\d{1,3}){3}$/.test(String(item.ip)) &&
                    !String(item.ip).startsWith('127.'),
            );
            if (local) return String(local.ip);
            const host = String(window.location?.hostname || '');
            return /^\d{1,3}(?:\.\d{1,3}){3}$/.test(host) &&
                !host.startsWith('127.')
                ? host
                : '';
        },

        cudaJoinState() {
            const id = this.cudaJoin.joinId;
            return (this.cudaJoin.status?.join_keys || []).find(
                (item) => item.join_id === id,
            );
        },

        cudaJoinCommandUsable() {
            const state = this.cudaJoinState();
            const expiresAt = Number(
                state?.expires_at || this.cudaJoin.expiresAt || 0,
            );
            return Boolean(
                this.cudaJoin.command &&
                (!state || state.status === 'pending') &&
                expiresAt * 1000 > Date.now(),
            );
        },

        cudaJoinStatusLabel() {
            if (!this.cudaJoin.command) return '';
            const state = this.cudaJoinState();
            if (state?.status === 'used') {
                return window.t('cluster.cuda.used');
            }
            if (state?.status === 'revoked') return window.t('cluster.cuda.revoked');
            const remaining = Math.ceil(
                (Number(state?.expires_at || this.cudaJoin.expiresAt || 0) *
                    1000 -
                    Date.now()) /
                    60000,
            );
            return remaining > 0
                ? window.t('cluster.cuda.expires_in').replace('{n}', String(remaining))
                : window.t('cluster.cuda.expired');
        },

        async loadCudaJoinStatus() {
            try {
                const payload = await this.apiFetch(CLUSTER_API.joinStatus, {
                    cache: 'no-store',
                });
                this.cudaJoin.status = payload || { join_keys: [], nodes: [] };
                const ids = this.cudaNodes().map((node) => node.node_id);
                if (!ids.includes(this.cudaFabricMemberA)) {
                    this.cudaFabricMemberA = ids[0] || '';
                }
                if (
                    !ids.includes(this.cudaFabricMemberB) ||
                    this.cudaFabricMemberB === this.cudaFabricMemberA
                ) {
                    this.cudaFabricMemberB =
                        ids.find((id) => id !== this.cudaFabricMemberA) || '';
                }
                this.cudaJoin.error = payload?.load_error || '';
            } catch (error) {
                this.cudaJoin.error =
                    error?.message || window.t('cluster.err.cuda_status');
            }
        },

        async revokeCudaJoinCommand() {
            const id = String(this.cudaJoin.joinId || '').trim();
            if (!id) return;
            try {
                await this.apiFetch(CLUSTER_API.revokeJoinKey(id), {
                    method: 'DELETE',
                });
                this.cudaJoin.command = '';
                this.cudaJoin.joinId = '';
                this.cudaJoin.expiresAt = 0;
                await this.loadCudaJoinStatus();
            } catch (error) {
                if (error?.status === 404) {
                    this.cudaJoin.command = '';
                    this.cudaJoin.joinId = '';
                    await this.loadCudaJoinStatus();
                    return;
                }
                this.cudaJoin.error =
                    error?.message || window.t('cluster.err.revoke_join');
            }
        },

        async generateCudaJoinCommand() {
            if (this.cudaJoin.loading) return;
            const controllerIp = this.cudaJoinSuggestedIp();
            if (!controllerIp) {
                this.cudaJoin.error =
                    window.t('cluster.cuda.controller_ip_required');
                return;
            }
            this.cudaJoin.controllerIp = controllerIp;
            this.cudaJoin.loading = true;
            this.cudaJoin.error = '';
            try {
                if (this.cudaJoinCommandUsable()) {
                    await this.revokeCudaJoinCommand();
                    if (this.cudaJoin.error) return;
                }
                const secure = window.location?.protocol === 'https:';
                const port = Number(window.location?.port) || (secure ? 443 : 80);
                const result = await this.apiFetch(CLUSTER_API.joinKeys, {
                    method: 'POST',
                    cache: 'no-store',
                    body: JSON.stringify({
                        controller_ip: controllerIp,
                        controller_port: port,
                        scheme: secure ? 'https' : 'http',
                        ttl_seconds: 1800,
                    }),
                });
                this.cudaJoin.command = result?.command || '';
                this.cudaJoin.joinId = result?.join_id || '';
                this.cudaJoin.expiresAt = Number(result?.expires_at || 0);
                await this.loadCudaJoinStatus();
            } catch (error) {
                this.cudaJoin.error =
                    error?.message || window.t('cluster.err.cuda_join');
            } finally {
                this.cudaJoin.loading = false;
            }
        },

        copyCudaJoinCommand() {
            const command = this.cudaJoin.command;
            if (!command) return;
            const done = () => {
                this.cudaJoin.copied = true;
                setTimeout(() => (this.cudaJoin.copied = false), 2000);
            };
            if (navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(command).then(done, done);
            } else {
                done();
            }
        },

        async verifyCudaFabric() {
            const selected = new Set([
                this.cudaFabricMemberA,
                this.cudaFabricMemberB,
            ]);
            const nodes = this.cudaNodes().filter((node) =>
                selected.has(node.node_id),
            );
            if (nodes.length !== 2 || this.cudaFabricLoading) return;
            this.cudaFabricLoading = true;
            this.cudaFabricError = '';
            this.cudaFabricResult = null;
            try {
                this.cudaFabricResult = await this.apiFetch(
                    CLUSTER_API.cudaFabricVerify,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            hosts: nodes.map((node) => ({
                                node_id: node.node_id,
                                ssh: node.ssh,
                            })),
                        }),
                    },
                );
                this.notify('success', window.t('cluster.toast.cuda_verified'));
                await this.refreshDevices();
            } catch (error) {
                this.cudaFabricError =
                    error?.message || window.t('cluster.err.cuda_verify');
            } finally {
                this.cudaFabricLoading = false;
            }
        },

        async refreshRdmaLinks() {
            if (this.rdmaLinks.loading) return;
            this.rdmaLinks.loading = true;
            try {
                this.rdmaLinks.data = await this.apiFetch(CLUSTER_API.rdmaLinks);
                this.rdmaLinks.error = '';
            } catch (error) {
                this.rdmaLinks.error =
                    error?.message || window.t('cluster.err.rdma_links');
            } finally {
                this.rdmaLinks.loading = false;
            }
        },

        async verifyRdmaLink(name) {
            if (this.rdmaLinks.verifying) return;
            this.rdmaLinks.verifying = name;
            this.rdmaLinks.error = '';
            try {
                const result = await this.apiFetch(CLUSTER_API.rdmaLinkVerify, {
                    method: 'POST',
                    body: JSON.stringify({ link: name }),
                });
                if (result.verified) {
                    this.notify('success', window.t('cluster.toast.rdma_verified'));
                } else {
                    this.rdmaLinks.error = result.reason || window.t('cluster.err.rdma_verify');
                }
                await this.refreshRdmaLinks();
            } catch (error) {
                this.rdmaLinks.error =
                    error?.message || window.t('cluster.err.rdma_verify');
            } finally {
                this.rdmaLinks.verifying = '';
            }
        },

        rdmaLinkDetail(link) {
            const measured = link.verification?.measurements;
            if (!link.verified || !measured) {
                return link.stale_reason || link.reason || '';
            }
            return window.t('cluster.rdma.measured')
                .replace('{latency}', measured.latency_p50_us.toFixed(1))
                .replace('{to}', measured.to_peer_gbit_s.toFixed(1))
                .replace('{from}', measured.from_peer_gbit_s.toFixed(1));
        },

        async downloadClusterDiagnostics() {
            if (this.diagnosticsLoading) return;
            this.diagnosticsLoading = true;
            this.diagnosticsError = '';
            try {
                const report = await this.apiFetch(CLUSTER_API.diagnostics, {
                    cache: 'no-store',
                });
                const blob = new Blob(
                    [JSON.stringify(report, null, 2) + '\n'],
                    { type: 'application/json' },
                );
                const link = document.createElement('a');
                const url = URL.createObjectURL(blob);
                link.href = url;
                link.download = `omlx-cluster-diagnostics-${new Date()
                    .toISOString()
                    .replaceAll(':', '-')}.json`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            } catch (error) {
                this.diagnosticsError =
                    error?.message || window.t('cluster.err.diagnostics');
            } finally {
                this.diagnosticsLoading = false;
            }
        },

        // =====================================================================
        // Assistant — the minimal Screen-3 slice
        // (the first UI slice notes)
        //
        // One card inside the plan step drives whichever flow the recorded
        // entry decision allows: measure the selected model (model-bound),
        // recommend from the model list, re-measure only if the recommendation
        // names another model, plan, apply, start, then poll the operation.
        //
        // It renders the backend's answer. It never computes a plan, a rank
        // order, a coordinator or an admission; it never launches, stages,
        // deploys or talks SSH; and every failure shows exactly one action
        // taken from error.action.
        // =====================================================================
        assistantState() {
            return this.assistant.state;
        },

        assistantRunning() {
            return (
                this.assistant.busy ||
                ['measuring', 'recommending', 'measuring_model'].includes(
                    this.assistant.state,
                )
            );
        },

        // The assistant card is mounted independently of wizardState(): the
        // entry must stay reachable on a Mac that already has a saved setup.
        // Show it once there is a paired peer to assist, or any assistant
        // state exists.
        assistantSurfaceVisible() {
            return Boolean(
                this.pairedDevices().length ||
                    this.assistant.measurementId ||
                    this.assistant.plan,
            );
        },

        assistantStateLabel() {
            const labels = {
                idle: window.t('cluster.assistant.state.idle'),
                measuring: window.t('cluster.assistant.state.measuring'),
                recommending: window.t('cluster.assistant.state.recommending'),
                measuring_model: window.t(
                    'cluster.assistant.state.measuring_model',
                ),
                planning: window.t('cluster.assistant.state.planning'),
                failed: window.t('cluster.assistant.state.failed'),
            };
            return labels[this.assistant.state] || '';
        },

        // Lightning MTP is a plan input, not a display preference: the draft
        // head has to fit beside rank 0's layers, so the plan and the
        // measurement that feeds it must agree on it. The catalogue says
        // whether this checkpoint carries the head at all; a model without one
        // would waste rank-0 space for weights that do not exist.
        assistantMtpFor(modelPath) {
            const wanted = modelPath || this.selectedModelPath;
            const model = this.modelOptions.find(
                (item) => item?.model_path === wanted,
            );
            return model?.mtp_compatible === true;
        },

        assistantModelLabel() {
            if (this.assistant.selectedModelLabel) {
                return this.assistant.selectedModelLabel;
            }
            return this.displayModelName({
                model_path:
                    this.assistant.selectedModel || this.selectedModelPath,
            });
        },

        // The assistant's ``selected_model`` is the planner path, while the
        // endpoint advertises the catalogue's serving id. The catalogue carries
        // both, so the handoff resolves the id here instead of inventing one.
        // An unresolved id links without a ``model`` parameter; it never falls
        // back to a different model silently.
        // ``/chat`` is a page of the server that owns the endpoint. When that
        // Mac is this Mac the browser already talks to it, so a relative link
        // keeps the session; when it is another Mac the link is that Mac's own
        // page, because the origin has no key and must not proxy.
        // One line, only when the backend measured it. No rate is ever derived
        // from the readiness canary or a prediction.
        assistantMacCount() {
            const plan = this.assistant.plan;
            if (Array.isArray(plan?.rank_order) && plan.rank_order.length) {
                return plan.rank_order.length;
            }
            const chosen = this.assistantRecommendedCandidate();
            if (chosen?.nodes_required) return chosen.nodes_required;
            return (this.selfDevice() ? 1 : 0) + this.pairedDevices().length;
        },

        // ---- Phase A: the chosen subset and the resource levers -----------
        assistantSelectedNodeIds() {
            const policyIds = this.assistant.plan?.selected_node_ids;
            if (Array.isArray(policyIds) && policyIds.length) return policyIds;
            const order = this.assistant.plan?.rank_order;
            return Array.isArray(order) ? order : [];
        },

        assistantNodeName(nodeId) {
            const device = this.allDevices().find(
                (item) => item.node_id === nodeId,
            );
            return device ? this.deviceName(device) : nodeId;
        },

        assistantSelectionLabel() {
            const selected = this.assistantSelectedNodeIds();
            if (!selected.length) return '';
            const names = selected.map((id) => this.assistantNodeName(id));
            const measured =
                this.assistant.measurementNodes.length ||
                (this.selfDevice() ? 1 : 0) + this.pairedDevices().length;
            const excluded = this.assistant.plan?.excluded_nodes || [];
            const excludedNames = excluded.map(
                (item) => this.assistantNodeName(item.node_id),
            );
            let text = `${names.join(' + ')} (${selected.length} of ${measured}`;
            if (excludedNames.length) {
                text += `; ${excludedNames.join(' and ')} not needed`;
            }
            return text + ')';
        },

        // The resolved policy is the only source for these labels: a value the
        // backend did not choose is never rendered. A chosen tradeoff (the
        // experts-resident alternative) is the policy that apply will use, so
        // the labels follow the selection.
        assistantResourcePolicy() {
            return (
                this.assistant.selectedTradeoff?.resource_policy ||
                this.assistant.plan?.resource_policy ||
                null
            );
        },

        // The plan or the chosen tradeoff: which signature/coordinator/order
        // apply and start must use. A tradeoff carries its own signature.
        assistantTradeoffs() {
            return this.assistant.plan?.tradeoffs || [];
        },

        assistantTradeoffSelected() {
            return Boolean(this.assistant.selectedTradeoff);
        },

        assistantTierLabel() {
            const policy = this.assistantResourcePolicy();
            if (!policy) return '';
            const tier = policy.memory_guard_tier || 'saved';
            return tier.charAt(0).toUpperCase() + tier.slice(1);
        },

        assistantPolicyContextLabel() {
            const policy = this.assistantResourcePolicy();
            if (!policy) return '';
            const tokens = Number(policy.context_tokens || 0);
            if (!tokens) return '';
            const target = Number(policy.target_context_tokens || 0);
            const value = `${tokens.toLocaleString()} tokens`;
            if (target > tokens) {
                return `${value} (reduced from ${target.toLocaleString()})`;
            }
            return value;
        },

        assistantOffloadLabel() {
            const policy = this.assistantResourcePolicy();
            if (!policy || policy.table_offload === 'none') return '';
            if (policy.table_offload === 'ple') return 'PLE table on SSD';
            if (policy.table_offload === 'engram') return 'Engram table on SSD';
            return '';
        },

        assistantExpertsLabel() {
            const policy = this.assistantResourcePolicy();
            if (!policy || !policy.moe_expert_offload) return '';
            const fraction = Number(policy.moe_resident_fraction || 0);
            if (!Number.isFinite(fraction) || fraction <= 0) return '';
            return `${Math.round(fraction * 100)} % resident`;
        },

        assistantContextLabel() {
            const tokens = this.assistant.selectedContextTokens ||
                this.targetContextTokens;
            return this.contextReservationLabel(tokens);
        },

        assistantCoordinatorName() {
            const nodeId = this.assistant.plan?.coordinator_node || '';
            if (!nodeId) return '';
            const device = this.allDevices().find(
                (item) => item.node_id === nodeId,
            );
            return device ? this.deviceName(device) : nodeId;
        },

        assistantCoordinatorInstruction() {
            return window
                .t('cluster.assistant.err.retry_coordinator')
                .replaceAll('{node}', this.assistantCoordinatorName());
        },

        assistantErrorMessage() {
            return (
                this.assistant.error?.message ||
                window.t('cluster.assistant.err.generic')
            );
        },

        // Screen 4 — the six observable rows, rendered from values the backend
        // already reports. The row that is current is the *last* one a backend
        // value named: a failure keeps it, an unknown phase keeps it, and no
        // timer, interval or optimistic step ever moves it.
        // The backend's own sentence under the current row. A known phase that
        // belongs to another row never labels this one; an unknown phase is
        // shown as-is, because it is the only thing the coordinator said.
        // Staging is the only phase that reports per-Mac counts. "2 of 3 Macs"
        // is a count the backend reported; it is never turned into a
        // percentage.
        assistantPlanWarnings() {
            const warnings = this.assistant.plan?.warnings || [];
            return Array.isArray(warnings) ? warnings : [];
        },

        assistantRankOrderLabel() {
            const order = this.assistant.plan?.rank_order;
            return Array.isArray(order) ? order.join(' → ') : '';
        },

        assistantStages() {
            const stages = this.assistant.plan?.stages;
            return Array.isArray(stages) ? stages : [];
        },

        assistantHasDetails() {
            return Boolean(this.assistant.measurementId || this.assistant.plan);
        },
        assistantRecommendedCandidate() {
            const response = this.assistant.recommendation;
            const candidates = Array.isArray(response?.candidates)
                ? response.candidates
                : [];
            const selectedId = response?.selected_model || '';
            return (
                candidates.find(
                    (item) => item.model_id === selectedId && item.feasible,
                ) ||
                candidates.find((item) => item.model_id === selectedId) ||
                candidates.find((item) => item.feasible) ||
                null
            );
        },

        // The Macs the measurement covers. This is derived from the pairing
        // snapshot — never typed by the user and never a rank decision.
        assistantNodes() {
            const nodes = [];
            const self = this.selfDevice();
            if (self?.node_id) {
                nodes.push({ node_id: self.node_id, ssh: '127.0.0.1' });
            }
            for (const peer of this.pairedDevices()) {
                if (!peer?.node_id) continue;
                nodes.push({
                    node_id: peer.node_id,
                    ssh: this.sshTargetFor(peer),
                    python_executable:
                        this.checks.probes[peer.node_id]?.result?.status
                            ?.runtime?.python_executable || undefined,
                });
            }
            return nodes;
        },

        // Macs below ASSISTANT_SMALL_MAC_RAM_GB cannot hold a workstation
        // reserve (>= 32 GiB) and still serve a stage: live, a 32 GB Mac
        // admitted 1.0 GiB and no model fitted. The assistant uses them as
        // Headless and says so; the backend enforces the same rule from
        // measured capacity.
        assistantRoles(nodes) {
            const selfId = this.selfDevice()?.node_id;
            const roles = {};
            for (const node of nodes) {
                const device = this.allDevices().find(
                    (item) => item.node_id === node.node_id,
                );
                const small =
                    this.deviceRamGb(device) > 0 &&
                    this.deviceRamGb(device) < ASSISTANT_SMALL_MAC_RAM_GB;
                roles[node.node_id] = small
                    ? 'headless'
                    : this.nodeRole(node.node_id, node.node_id === selfId);
            }
            return roles;
        },

        // One plain sentence per small Mac, so nobody wonders why it changed.
        assistantSmallMacNotes() {
            const selfId = this.selfDevice()?.node_id;
            return this.assistantNodes()
                .map((node) =>
                    this.allDevices().find((item) => item.node_id === node.node_id),
                )
                .filter((device) => {
                    const ram = this.deviceRamGb(device);
                    return ram > 0 && ram < ASSISTANT_SMALL_MAC_RAM_GB;
                })
                .map((device) =>
                    window
                        .t('cluster.assistant.small_mac_note')
                        .replace('{node}', this.deviceName(device))
                        .replace('{gb}', String(Math.round(this.deviceRamGb(device)))),
                );
        },

        assistantCandidates() {
            const models = this.modelOptions.filter((model) => model?.model_path);
            // The backend accepts at most ASSISTANT_MAX_CANDIDATES candidates.
            // A larger catalogue is narrowed deterministically instead of
            // being refused: the chosen model first, then models a measured
            // Mac already holds (no staging), then the smallest remaining.
            const chosen = this.assistant.selectedModel || this.selectedModelPath;
            const held = new Set();
            for (const node of this.assistant.measurementNodes || []) {
                for (const path of node?.model_paths || []) held.add(path);
            }
            const rank = (model) =>
                model.model_path === chosen ? 0 : held.has(model.model_path) ? 1 : 2;
            return models
                .map((model, index) => ({ model, index }))
                .sort((a, b) => {
                    const byRank = rank(a.model) - rank(b.model);
                    if (byRank !== 0) return byRank;
                    if (rank(a.model) === 2) {
                        const bySize =
                            (Number(a.model.estimated_size) || 0) -
                            (Number(b.model.estimated_size) || 0);
                        if (bySize !== 0) return bySize;
                    }
                    return a.index - b.index;
                })
                .slice(0, ASSISTANT_MAX_CANDIDATES)
                .map(({ model }) => ({
                    model_path: model.model_path,
                    friendly_name: this.displayModelName(model),
                    declared_context_tokens:
                        this.assistantDeclaredContext(model),
                }));
        },

        assistantDeclaredContext(model) {
            const tokens = Number(model?.model_context_length || 0);
            if (!tokens) return undefined;
            return Math.min(1_048_576, Math.max(1, Math.round(tokens)));
        },

        // -----------------------------------------------------------------
        // Search — the automatic constellation loop (POST /assistant/search)
        //
        // An action on this card, not a new state-machine value: the backend
        // ranks, deploys and measures several constellations, keeps the best
        // measured and stops the losers, then answers once. The request is
        // synchronous and can take minutes, so it is single-flight (busy flag)
        // and generation-guarded exactly like the polls; a stale answer is
        // dropped. A future async-job refactor is the proper long-running model.
        // -----------------------------------------------------------------
        // Only a measured constellation has a rate; an unmeasured one never
        // shows a number (never invent a rate). Reuses the observed.* wording.
        // -----------------------------------------------------------------
        // Tuning — the winning deployment's own knobs (AGENTS.md the notes "Tuning is
        // a saved preset, not a re-tune on every Start")
        //
        // Two actions on this card, never new state-machine values:
        //  * Fine-tune (POST /tune) sweeps samplers per request on the running
        //    cluster. It is quick and the cluster keeps serving.
        //  * Tune MTP depth (POST /tune-redeploy) redeploys every MTP switch /
        //    depth config and measures each one for real. It stops the running
        //    cluster first and every config afterwards, so it ends with the
        //    cluster stopped; the existing Start again applies the saved preset.
        //
        // Every number rendered here is a measured rate from the backend's own
        // answer. A knob the backend calls not applicable is rendered as such,
        // and a config that measured nothing says "not measured" — the card
        // never invents a rate.
        // -----------------------------------------------------------------
        // Only a rate the backend really measured is a rate: null, 0 and
        // unparsable values are "not measured", never 0.0 tok/s.
        // One job, one loop, one answer at a time: GET the job, apply the
        // snapshot, stop at completed/failed. A snapshot from an older
        // generation never lands.
        // -----------------------------------------------------------------
        // Run on this Mac — single-node inference, no cluster
        //
        // the live-run notes measured it: a model that fits on one Mac is
        // faster served by that Mac's own engine (254 tok/s) than by a pipeline
        // (53 tok/s on the 4-Mac cluster). The recommendation names that option
        // (`single_node`); when it names *this* Mac, the card offers one button.
        //
        // It is an action with its own async job, not a new assistant state: the
        // browser POSTs once and polls the job (single-flight, generation
        // guarded), and every line it shows is a field the backend reported. No
        // token rate, percentage or byte count is invented, and a failure keeps
        // its one action.
        // -----------------------------------------------------------------
        // This flow is local by contract: it can only run on the Mac the
        // browser is already talking to. A single-node answer that names
        // another Mac is not offered here — that Mac's own page may offer it.
        // Stop/Unload is offered while the job holds this Mac's engine: a
        // queued or running load, or a ready one that is still serving. A
        // stopped job is never stopped twice, and one stop is in flight at a
        // time, so a double click cannot post the same request again.
        // The load is narrated until it settles, including the readiness
        // canary the backend runs before it says ready — the spinner is driven
        // by the job's own state, not only by the promise that started it.
        // The backend's own sentence ("loading … on this Mac"), never an
        // invented percentage.
        // Ready only: a URL nobody answers is worse than none.
        // Screen 5, model-only: this Mac is the one serving, so the link is this
        // Mac's existing /chat with the job's serving id. An unresolved id links
        // without the parameter and never guesses.
        // One job, one loop: a snapshot from an older generation never lands,
        // the loop stops at ready/failed, and a stop retires it so a late
        // snapshot cannot resurrect a job this Mac no longer serves.
        // Stop/Unload for "Run on this Mac". One POST to the job's own route,
        // then one re-read: the job's last word decides whether this Mac
        // stopped serving. A stop that fails keeps the serving surface (the
        // engine is still loaded) and its one action; a stopped job retires
        // the endpoint and the chat link. Idempotent — a second click posts
        // nothing, and the shell route itself is idempotent too.
        // The job id stays so a repeated stop is still idempotent, but no
        // serving surface survives: no endpoint, no chat link, no message.
        // ---- single-node tuning (sampler + MTP on/off/depth) --------------

        // -----------------------------------------------------------------
        // Flow
        // -----------------------------------------------------------------
        async startAssistant() {
            if (this.assistantRunning()) return null;
            // A fresh run starts without the previous run's fallback consent.
            this.assistant.confirmTwoMacFallback = false;
            this.assistant.topologyFallback = null;
            this.assistant.missingLinks = [];
            return this.runAssistantMeasure();
        },

        // The ``copy_model`` action selects the candidate and runs the normal
        // assistant flow; apply's staging is the source-first copy.
        async startAssistantForModel(modelPath) {
            if (!modelPath || this.assistantRunning()) {
                return this.reopenAssistantPicker();
            }
            this.assistant.selectedModel = modelPath;
            this.selectedModelPath = modelPath;
            return this.startAssistant();
        },

        resetAssistant() {
            this.assistant.measurementId = '';
            this.assistant.measurementModel = '';
            this.assistant.recommendation = null;
            this.assistant.selectedModel = '';
            this.assistant.selectedModelId = '';
            this.assistant.selectedModelLabel = '';
            this.assistant.selectedContextTokens = null;
            this.assistant.plan = null;
            this.assistant.selectedTradeoff = null;
            this.assistant.inputs = null;
            this.assistant.measurementNodes = [];
            this.assistant.measurePollErrors = 0;
            this.assistant.measurePollSkip = 0;
            this.assistant.freeSpaceOpen = false;
            this.assistant.freeSpaceNode = '';
            this.assistant.freeSpaceConfirm = '';
            this.assistant.freeSpaceBusy = false;
            this.assistant.freeSpaceError = '';
            this.assistant.warnings = [];
            this.assistant.error = null;
            this.assistant.copied = '';
            this.assistant.state = 'idle';
        },

        async runAssistantMeasure() {
            this.stopAssistantPolls();
            const generation = ++this.assistant.generation;
            this.resetAssistant();
            this.assistant.state = 'measuring';
            this.assistant.phase = 'recommend';
            this.assistant.busy = true;
            try {
                const nodes = this.assistantNodes();
                if (!nodes.length) {
                    this.failAssistant(
                        'no_nodes',
                        window.t('cluster.assistant.err.no_nodes'),
                        'pair_mac',
                    );
                    return;
                }
                // Hardware-only entry measurement (Phase C3): the model is not
                // chosen yet. The recommendation then measures the chosen
                // model again before plan.
                const handle = await this.apiFetch(ASSISTANT_API.measure, {
                    method: 'POST',
                    body: JSON.stringify({
                        nodes,
                        roles: this.assistantRoles(nodes),
                    }),
                });
                if (generation !== this.assistant.generation) return;
                this.assistant.measurementId = handle?.measurement_id || '';
                this.assistant.measurementModel = '';
                this.startAssistantMeasurePoll(generation);
            } catch (error) {
                if (generation !== this.assistant.generation) return;
                this.failAssistantFrom(error, 'measure');
            } finally {
                if (generation === this.assistant.generation) {
                    this.assistant.busy = false;
                }
            }
        },

        async runAssistantRecommend(generation) {
            if (generation !== this.assistant.generation) return;
            this.assistant.state = 'recommending';
            this.assistant.busy = true;
            try {
                const candidates = this.assistantCandidates();
                if (!candidates.length) {
                    this.failAssistant(
                        'no_model',
                        window.t('cluster.assistant.err.no_model'),
                        'diagnostics',
                    );
                    return;
                }
                const response = await this.apiFetch(
                    ASSISTANT_API.recommend,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            measurement_id: this.assistant.measurementId,
                            candidates,
                            context_tokens: Math.max(
                                1,
                                Number(this.targetContextTokens) || 32768,
                            ),
                            execution_profile: this.executionProfile,
                        }),
                    },
                );
                if (generation !== this.assistant.generation) return;
                this.applyAssistantRecommendation(response);
                await this.continueAfterRecommendation(generation);
            } catch (error) {
                if (generation !== this.assistant.generation) return;
                this.failAssistantFrom(error, 'measure');
            } finally {
                if (generation === this.assistant.generation) {
                    this.assistant.busy = false;
                }
            }
        },

        applyAssistantRecommendation(response) {
            this.assistant.recommendation = response || null;
            const chosen = this.assistantRecommendedCandidate();
            const context = Number(response?.selected_context_tokens || 0);
            this.assistant.selectedModel = chosen?.model_path || '';
            this.assistant.selectedModelId =
                chosen?.model_id || response?.selected_model || '';
            this.assistant.selectedModelLabel =
                chosen?.label ||
                this.displayModelName({
                    model_path: chosen?.model_path || '',
                });
            this.assistant.selectedContextTokens = context > 0 ? context : null;
            // Keep the existing picker in sync so the primary path and the
            // advanced controls describe the same choice.
            if (chosen?.model_path) this.selectedModelPath = chosen.model_path;
            if (context > 0) this.targetContextTokens = context;
        },

        async continueAfterRecommendation(generation) {
            if (generation !== this.assistant.generation) return;
            const target =
                this.assistant.selectedModel || this.selectedModelPath;
            if (!target) {
                this.failAssistant(
                    'no_model',
                    window.t('cluster.assistant.err.no_model'),
                    'diagnostics',
                );
                return;
            }
            // ``plan`` keeps its strict model-bound measurement match, so a
            // recommendation that names another model is measured again first.
            if (this.assistant.measurementModel === target) {
                await this.runAssistantPlan(generation);
                return;
            }
            await this.runAssistantModelMeasure(generation, target);
        },

        async runAssistantModelMeasure(generation, modelPath) {
            if (generation !== this.assistant.generation) return;
            this.assistant.state = 'measuring_model';
            this.assistant.phase = 'plan';
            this.assistant.busy = true;
            try {
                const nodes = this.assistantNodes();
                const handle = await this.apiFetch(ASSISTANT_API.measure, {
                    method: 'POST',
                    body: JSON.stringify({
                        nodes,
                        roles: this.assistantRoles(nodes),
                        model_path: modelPath,
                        mtp: this.assistantMtpFor(modelPath),
                    }),
                });
                if (generation !== this.assistant.generation) return;
                this.assistant.measurementId = handle?.measurement_id || '';
                this.assistant.measurementModel = modelPath;
                this.startAssistantMeasurePoll(generation);
            } catch (error) {
                if (generation !== this.assistant.generation) return;
                this.failAssistantFrom(error, 'measure');
            } finally {
                if (generation === this.assistant.generation) {
                    this.assistant.busy = false;
                }
            }
        },

        // Frozen plan-time inputs: identical object for plan and apply, so a
        // value cannot drift between the two requests.
        assistantPlanInputs() {
            const modelPath =
                this.assistant.selectedModel || this.selectedModelPath;
            if (!modelPath || !this.assistant.measurementId) return null;
            return {
                measurement_id: this.assistant.measurementId,
                model_path: modelPath,
                context_tokens: Math.max(
                    1,
                    Number(
                        this.assistant.selectedContextTokens ||
                            this.targetContextTokens,
                    ) || 32768,
                ),
                mtp: this.assistantMtpFor(modelPath),
                rank0_extra_bytes: 0,
                execution_profile: this.executionProfile,
                microbatch_size: null,
                // Phase A: the saved settings are the default; the solver may
                // propose a lever and the plan shows it. Only "Use all Macs"
                // is an explicit user override here.
                node_selection: this.assistant.nodeSelection || 'minimal',
                confirm_two_mac_fallback: !!this.assistant.confirmTwoMacFallback,
                admission: 'p50',
            };
        },

        // Phase A advanced override: re-plan with the other node selection.
        // Only while no job is running and no cluster is up — re-planning a
        // running cluster is out of scope (Screen 6), and a second concurrent
        // plan would race the flow's own generation-guarded plan.
        assistantCanChangeNodeSelection() {
            return (
                Boolean(this.assistant.measurementId) &&
                !this.assistantRunning() &&
                this.assistant.state !== 'ready'
            );
        },

        toggleAssistantNodeSelection() {
            if (!this.assistantCanChangeNodeSelection()) return;
            this.assistant.nodeSelection =
                this.assistant.nodeSelection === 'all' ? 'minimal' : 'all';
            // A fresh generation retires any in-flight answer from the old
            // selection before the new plan is requested.
            const generation = ++this.assistant.generation;
            return this.runAssistantPlan(generation);
        },

        async runAssistantPlan(generation = this.assistant.generation) {
            if (generation !== this.assistant.generation) return;
            const inputs = this.assistantPlanInputs();
            if (!inputs) {
                this.failAssistant(
                    'plan_inputs_missing',
                    window.t('cluster.assistant.err.plan_inputs'),
                    'measure',
                );
                return;
            }
            this.stopAssistantPolls();
            this.assistant.state = 'planning';
            this.assistant.busy = true;
            this.assistant.error = null;
            this.assistant.plan = null;
            this.assistant.topologyFallback = null;
            this.assistant.missingLinks = [];
            try {
                const plan = await this.apiFetch(ASSISTANT_API.plan, {
                    method: 'POST',
                    body: JSON.stringify(inputs),
                });
                if (generation !== this.assistant.generation) return;
                this.assistant.plan = plan || null;
                this.assistant.selectedTradeoff = null;
                // The backend returns the salted fallback plan only when the
                // consent was sent. Keep the flag exactly as the plan used it,
                // so the shown plan is the one the signature names.
                const usedFallback = !!plan?.topology?.fallback_used;
                this.assistant.confirmTwoMacFallback = usedFallback;
                this.assistant.inputs = {
                    ...inputs,
                    confirm_two_mac_fallback: usedFallback,
                };
                this.assistant.warnings = plan?.warnings || [];
            } catch (error) {
                if (generation !== this.assistant.generation) return;
                this.failAssistantFrom(error, 'plan');
            } finally {
                if (generation === this.assistant.generation) {
                    this.assistant.busy = false;
                }
            }
        },

        // The offered 2-Mac fallback is never applied from the refusal body:
        // the user clicked it, so re-plan with the consent flag and let the
        // backend return the plan whose signature apply will verify.
        async confirmAssistantTwoMacFallback() {
            if (!this.assistant.topologyFallback || this.assistantRunning()) {
                return null;
            }
            this.assistant.confirmTwoMacFallback = true;
            this.assistant.nodeSelection = 'minimal';
            this.assistant.error = null;
            const generation = ++this.assistant.generation;
            return this.runAssistantPlan(generation);
        },

        // -----------------------------------------------------------------
        // Single-flight poll loop (one measurement, generation-guarded)
        // -----------------------------------------------------------------
        startAssistantMeasurePoll(generation) {
            if (generation !== this.assistant.generation) return;
            this.stopAssistantMeasurePoll();
            this.assistant.measurePollErrors = 0;
            this.assistant.measurePollSkip = 0;
            this.assistant.measureTimer = setInterval(
                () => this.pollAssistantMeasurement(),
                CLUSTER_POLL_MS,
            );
        },

        stopAssistantMeasurePoll() {
            if (this.assistant.measureTimer) {
                clearInterval(this.assistant.measureTimer);
                this.assistant.measureTimer = null;
            }
        },

        stopAssistantPolls() {
            this.stopAssistantMeasurePoll();
        },

        async pollAssistantMeasurement() {
            if (this.assistant.measurePollBusy) return;
            if (this.assistant.measurePollSkip > 0) {
                this.assistant.measurePollSkip -= 1;
                return;
            }
            const generation = this.assistant.generation;
            const measurementId = this.assistant.measurementId;
            if (!measurementId) {
                this.stopAssistantMeasurePoll();
                return;
            }
            let snapshot;
            this.assistant.measurePollBusy = true;
            try {
                snapshot = await this.apiFetch(
                    `${ASSISTANT_API.status}?measurement_id=${encodeURIComponent(measurementId)}`,
                );
            } catch (error) {
                if (generation !== this.assistant.generation) return;
                // A transient network error must not kill a minutes-long
                // measurement: retry a few times with backoff, timer alive.
                this.assistant.measurePollErrors += 1;
                if (this.assistant.measurePollErrors < ASSISTANT_POLL_MAX_ERRORS) {
                    this.assistant.measurePollSkip =
                        2 ** (this.assistant.measurePollErrors - 1);
                    return;
                }
                this.stopAssistantMeasurePoll();
                this.failAssistantFrom(error, 'measure');
                return;
            } finally {
                this.assistant.measurePollBusy = false;
            }
            this.assistant.measurePollErrors = 0;
            if (
                generation !== this.assistant.generation ||
                measurementId !== this.assistant.measurementId
            ) {
                return;
            }
            const state = String(snapshot?.state || '');
            if (state === 'ready') {
                this.stopAssistantMeasurePoll();
                this.assistant.measurementModel =
                    snapshot?.model || this.assistant.measurementModel;
                // The measured Macs, each with the models its own catalogue
                // listed. The free-space affordance may only show a Mac the
                // user picked from this list.
                this.assistant.measurementNodes = Array.isArray(snapshot?.nodes)
                    ? snapshot.nodes
                    : [];
                await this.afterAssistantMeasurement(generation);
                return;
            }
            if (state === 'failed') {
                this.stopAssistantMeasurePoll();
                const error = snapshot?.error || {};
                this.failAssistant(
                    error.code || 'measurement_failed',
                    error.message ||
                        window.t('cluster.assistant.err.measure_failed'),
                    error.action || 'measure',
                );
            }
        },

        async afterAssistantMeasurement(generation) {
            if (generation !== this.assistant.generation) return;
            if (this.assistant.phase === 'plan') {
                await this.runAssistantPlan(generation);
                return;
            }
            await this.runAssistantRecommend(generation);
        },

        // -----------------------------------------------------------------
        // Failures: exactly one action, taken from error.action
        // -----------------------------------------------------------------
        failAssistant(code, message, action) {
            this.stopAssistantPolls();
            this.assistant.busy = false;
            this.assistant.state = 'failed';
            this.assistant.error = {
                code: String(code || 'request_failed'),
                message: String(
                    message || window.t('cluster.assistant.err.generic'),
                ),
                action: String(action || 'diagnostics'),
            };
        },

        failAssistantFrom(error, fallbackAction) {
            const payload = error?.payload || {};
            // A 409 measurement_in_progress names the running measurement the
            // same way; without adopting it the "status"/"measure" recovery
            // had no ID and looped on the same refusal forever.
            if (payload.measurement_id) {
                this.assistant.measurementId = payload.measurement_id;
            }
            if (payload.code === 'topology_incomplete') {
                this.assistant.topologyFallback = payload.fallback || null;
                this.assistant.missingLinks = Array.isArray(payload.missing_links)
                    ? payload.missing_links
                    : [];
            }
            this.failAssistant(
                payload.code || 'request_failed',
                payload.message ||
                    error?.message ||
                    window.t('cluster.assistant.err.generic'),
                payload.action || fallbackAction || 'diagnostics',
            );
        },

        // -----------------------------------------------------------------
        // One action per failure — dispatch only, never a fallback selection
        // -----------------------------------------------------------------
        assistantActionLabel(action) {
            const known = [
                'measure',
                'plan',
                'status',
                'add_mac',
                'pair_mac',
                'copy_model',
                'configure_endpoint',
                'retry_coordinator',
                'diagnostics',
                'adjust_model_or_context',
                // The recommendation's own near-miss actions. They were already
                // part of its answer, but the card did not know them, so a
                // refusal followed by a role or model change left it showing
                // "Continue" with nothing behind the button.
                'choose_smaller_model',
                'choose_pipeline_model',
                'check_model_path',
                'free_disk_space',
            ];
            if (!known.includes(action)) {
                return window.t('cluster.assistant.action.unknown');
            }
            return window
                .t(`cluster.assistant.action.${action}`)
                .replace('{node}', this.assistantCoordinatorName());
        },

        assistantActionHandler(action) {
            // Double-click protection: the first action already set busy (or
            // started a poll); a second click must not post the same request
            // again and flip a running flow to failed on the 409 answer.
            if (this.assistant.busy) return;
            switch (action) {
                case 'measure':
                    // A refused measure that names a running measurement
                    // resumes it; resetting would POST the same measure into
                    // the same 409 forever.
                    if (
                        this.assistant.measurementId &&
                        this.assistant.error?.code === 'measurement_in_progress'
                    ) {
                        return this.resumeAssistantPolling();
                    }
                    return this.startAssistant();
                case 'plan':
                    return this.runAssistantPlan();
                case 'status':
                    return this.resumeAssistantPolling();
                case 'retry_coordinator':
                    return this.retryAssistantCoordinator();
                case 'pair_mac':
                case 'add_mac':
                    return this.openAssistantPairing();
                case 'configure_endpoint':
                case 'diagnostics':
                    // The redacted assistant bundle belongs to the run
                    // backend; this slice points at oMLX's own cluster
                    // diagnostics export, which already exists.
                    return this.downloadClusterDiagnostics();
                case 'adjust_model_or_context':
                // The recommendation names the next thing worth changing. Every
                // one of these means "change the model, the context or what
                // each Mac contributes, then start again", and the picker is
                // where those controls live. Returning the card to idle also
                // restores the entry button: without that a refusal was a dead
                // end that only a page reload could clear.
                case 'choose_smaller_model':
                case 'choose_pipeline_model':
                case 'check_model_path':
                    return this.reopenAssistantPicker();
                // ``free_disk_space`` is the one failure with a written step:
                // delete an unused model on the Mac that is short, then
                // measure again. The panel only ever lists a Mac the user
                // picked.
                case 'free_disk_space':
                    return this.openAssistantFreeSpace();
                // ``copy_model`` means "use this model": the assistant selects
                // it and runs the flow again. Staging the copy is the run
                // backend's job, source-first from a complete peer copy; the
                // browser never downloads or deletes anything.
                case 'copy_model': {
                    const candidate = this.assistantRecommendedCandidate();
                    return this.startAssistantForModel(
                        candidate?.model_path || this.assistant.selectedModel,
                    );
                }
                default:
                    return this.notify(
                        'info',
                        window.t('cluster.assistant.action.other'),
                    );
            }
        },

        // Retry on the same elected coordinator. Nothing here names this Mac:
        // the plan is re-requested for the same measured Macs.
        async retryAssistantCoordinator() {
            const generation = ++this.assistant.generation;
            this.assistant.error = null;
            if (this.assistant.inputs && this.assistant.plan) {
                await this.runAssistantPlan(generation);
                return;
            }
            await this.runAssistantMeasure();
        },

        // Resume the running measurement after a refusal that named it, and
        // clear the failure banner; without this the same 409 repeats forever
        // behind a stale "failed".
        resumeAssistantPolling() {
            const generation = this.assistant.generation;
            if (this.assistant.measurementId && !this.assistant.plan) {
                this.assistant.error = null;
                this.assistant.state =
                    this.assistant.phase === 'plan'
                        ? 'measuring_model'
                        : 'measuring';
                this.startAssistantMeasurePoll(generation);
            }
        },

        // Open the existing pairing surface. Pairing itself stays where it
        // already lives; nothing is measured, planned or started here.
        openAssistantPairing() {
            this.assistant.error = null;
            this.assistant.state = 'idle';
            const target =
                this.discoveredDevices()[0] ||
                this.pendingApprovals()[0] ||
                null;
            if (target) {
                this.membershipPanelOpen = Boolean(
                    this.configuredDeployment(),
                );
                this.beginPairing(target);
            } else {
                this.stage = null;
            }
            this.notify('info', window.t('cluster.assistant.pairing_opened'));
        },

        reopenAssistantPicker() {
            this.assistant.error = null;
            this.assistant.state = 'idle';
            this.notify('info', window.t('cluster.assistant.adjust_hint'));
        },

        // -----------------------------------------------------------------
        // Free space: the UI of the one action `free_disk_space`
        // (the live-run notes). Nothing here deletes by itself: the user picks
        // the Mac, then one model on it, then confirms.
        // -----------------------------------------------------------------
        assistantNodeLabel(nodeId) {
            const device = this.allDevices().find(
                (item) => item.node_id === nodeId,
            );
            return device ? this.deviceName(device) : nodeId;
        },

        assistantFreeSpaceNodes() {
            return Array.isArray(this.assistant.measurementNodes)
                ? this.assistant.measurementNodes.filter(
                      (node) => node?.node_id,
                  )
                : [];
        },

        openAssistantFreeSpace() {
            this.assistant.freeSpaceError = '';
            this.assistant.freeSpaceConfirm = '';
            this.assistant.freeSpaceNode = '';
            if (!this.assistantFreeSpaceNodes().length) {
                // Nothing was measured, so there is no Mac whose models could
                // be listed. The failure keeps its one action and the card
                // points at the controls instead of showing an empty panel.
                this.reopenAssistantPicker();
                return;
            }
            this.assistant.freeSpaceOpen = true;
        },

        chooseAssistantFreeSpaceNode(nodeId) {
            this.assistant.freeSpaceNode = nodeId;
            this.assistant.freeSpaceConfirm = '';
            this.assistant.freeSpaceError = '';
        },

        assistantFreeSpaceModels() {
            const node = this.assistantFreeSpaceNodes().find(
                (item) => item.node_id === this.assistant.freeSpaceNode,
            );
            const models = Array.isArray(node?.models) ? node.models : [];
            // The model the cluster is about to run is never offered: freeing
            // space must not delete the checkpoint the plan is for. The route
            // refuses it as well, so this is only the visible half of the rule.
            const planned = new Set(
                [
                    this.assistant.selectedModel,
                    this.assistant.selectedModelId,
                    this.selectedModelPath,
                ].filter(Boolean),
            );
            return models
                .filter(
                    (model) =>
                        model?.model_id &&
                        !planned.has(model.model_id) &&
                        !planned.has(model.model_path),
                )
                .map((model) => ({
                    model_id: model.model_id,
                    label: model.model_id,
                    size: this.assistantModelSize(model.size_bytes),
                }));
        },

        assistantModelSize(sizeBytes) {
            const gib = Number(sizeBytes || 0) / 1024 ** 3;
            if (!(gib > 0)) {
                return window.t('cluster.assistant.free_space.size_unknown');
            }
            return window
                .t('cluster.assistant.free_space.size')
                .replace('{gib}', gib.toFixed(gib >= 10 ? 0 : 1));
        },

        assistantDeleteLabel(model) {
            if (this.assistant.freeSpaceConfirm !== model.model_id) {
                return window.t('cluster.assistant.free_space.delete');
            }
            return window
                .t('cluster.assistant.free_space.delete_confirm')
                .replace('{model}', model.model_id);
        },

        async deleteAssistantModel(model) {
            if (this.assistant.freeSpaceBusy) return;
            if (this.assistant.freeSpaceConfirm !== model.model_id) {
                // Destructive: the first click only arms this row.
                this.assistant.freeSpaceConfirm = model.model_id;
                this.assistant.freeSpaceError = '';
                return;
            }
            const nodeId = this.assistant.freeSpaceNode;
            const measurementId = this.assistant.measurementId;
            if (!nodeId || !measurementId) return;
            this.assistant.freeSpaceBusy = true;
            this.assistant.freeSpaceError = '';
            try {
                await this.apiFetch(
                    `${ASSISTANT_API.models(nodeId, model.model_id)}?measurement_id=${encodeURIComponent(measurementId)}`,
                    { method: 'DELETE' },
                );
                this.notify(
                    'success',
                    window
                        .t('cluster.assistant.free_space.deleted')
                        .replace('{model}', model.model_id)
                        .replace('{node}', this.assistantNodeLabel(nodeId)),
                );
                // How much a delete really freed is that Mac's number, so the
                // assistant measures again instead of assuming the copy fits.
                this.assistant.freeSpaceOpen = false;
                await this.runAssistantMeasure();
            } catch (error) {
                const payload = error?.payload || {};
                this.assistant.freeSpaceError =
                    payload.message ||
                    error?.message ||
                    window.t('cluster.assistant.free_space.failed');
            } finally {
                this.assistant.freeSpaceBusy = false;
                this.assistant.freeSpaceConfirm = '';
            }
        },

        // -----------------------------------------------------------------
        // Reload resume: the last persisted deployment, no re-election.
        // -----------------------------------------------------------------
        // B3: one bounded read, so a probe that landed after the ready poll is
        // visible without a reload. It reuses the poll's applier so state and
        // progress stay in sync (a stale `loading` could otherwise survive a
        // ready snapshot), and the generation guard drops a late response. On
        // the ready surface it never starts anything.
        // -----------------------------------------------------------------
        // Ready-only affordances
        // -----------------------------------------------------------------
        // Screen 6: state + stop/restart, nothing more. The browser calls one
        // assistant route; the backend relays to the coordinator's own unload
        // route with that Mac's key. The endpoint is closed afterwards, so no
        // Chat/agent link survives a stop.
        // Re-apply the frozen in-memory plan inputs, then the existing poll
        // starts it. No re-planning, no model/context edit, no Mac changes.
        copyAssistantValue(value) {
            const text = String(value || '');
            if (!text) return;
            const done = () => {
                this.assistant.copied = text;
                setTimeout(() => {
                    if (this.assistant.copied === text) {
                        this.assistant.copied = '';
                    }
                }, 2000);
                this.notify('success', window.t('cluster.assistant.copied'));
            };
            // A rejected write or a missing clipboard is a failure, never a
            // silent success: only a real copy marks the value and toasts.
            const failed = () => {
                this.notify('error', window.t('cluster.assistant.err.generic'));
            };
            if (navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(text).then(done, failed);
            } else {
                failed();
            }
        },

        // =====================================================================
        // Feedback — toasts, never modals
        // =====================================================================
        splitDevices() {
            return this.pairedDevices().filter(
                (device) => !this.deviceHelperOnly(device),
            );
        },

        deviceHelperOnly(device) {
            return device?.caps?.control === 'agent';
        },

        pairingStepOne(device = this.pairing.target) {
            // A helper daemon has no screen; its code is in its log.
            const key = this.deviceHelperOnly(device)
                ? 'cluster.pairing.step1_helper'
                : 'cluster.pairing.step1';
            return window.t(key)
                .replaceAll('{name}', this.deviceName(device))
                .replaceAll('{self}', this.deviceName(this.selfDevice()));
        },

        cudaGuideMacAddress() {
            const host = window.location.hostname;
            if (/^\d+\.\d+\.\d+\.\d+$/.test(host) && !host.startsWith('127.')) {
                return host;
            }
            const ipv4 = (this.selfDevice()?.addrs || [])
                .map((addr) => addr.ip)
                .filter((ip) => /^\d+\.\d+\.\d+\.\d+$/.test(ip) && !ip.startsWith('127.'));
            return ipv4.find((ip) => !ip.endsWith('.1')) || ipv4[0] || 'MAC_ADDRESS';
        },

        cudaGuideSteps() {
            const image = 'jundot/omlx-cuda-helper';
            const run = 'sudo docker run -d --name omlx-helper --restart unless-stopped --gpus all '
                + `--network host -v omlx-helper:/data ${image}\nsudo docker logs -f omlx-helper`;
            if (this.cudaGuideOs === 'windows') {
                const port = this.selfDevice()?.http_port || window.location.port || 8000;
                return [
                    {
                        text: 'cluster.cuda_guide.windows_docker',
                        command: 'winget install -e --id Docker.DockerDesktop',
                    },
                    {
                        text: 'cluster.cuda_guide.windows_firewall',
                        command: 'New-NetFirewallRule -DisplayName "oMLX helper" -Direction Inbound '
                            + '-Protocol TCP -LocalPort 8850-8852 -Action Allow',
                    },
                    {
                        text: 'cluster.cuda_guide.windows_run',
                        command: 'docker run -d --name omlx-helper --restart unless-stopped --gpus all '
                            + `-p 8850-8852:8850-8852 -v omlx-helper:/data ${image} `
                            + `--no-mdns --pair ${this.cudaGuideMacAddress()}:${port}\n`
                            + 'docker logs -f omlx-helper',
                    },
                ];
            }
            if (this.cudaGuideOs === 'spark') {
                return [{ text: 'cluster.cuda_guide.spark_run', command: run }];
            }
            const keyring = '/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg';
            return [
                {
                    text: 'cluster.cuda_guide.linux_docker',
                    command: [
                        'curl -fsSL https://get.docker.com | sudo sh',
                        'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey'
                            + ` | sudo gpg --yes --dearmor -o ${keyring}`,
                        'curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list'
                            + ` | sed 's#deb https://#deb [signed-by=${keyring}] https://#g'`
                            + ' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list',
                        'sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit',
                        'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker',
                    ].join('\n'),
                },
                { text: 'cluster.cuda_guide.linux_run', command: run },
            ];
        },

        copyCudaGuide(command, key) {
            const done = () => {
                this.cudaGuideCopied = key;
                setTimeout(() => {
                    if (this.cudaGuideCopied === key) this.cudaGuideCopied = '';
                }, 2000);
            };
            if (navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(command).then(done, done);
            } else {
                done();
            }
        },

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
