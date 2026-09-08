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
//     DELETE /api/cluster/devices/{node_id}     — unpair
//   STUB (defined by Module C, to be implemented in Module A's probe path):
//     GET    /api/cluster/discovery/health      — {multicast_rx_within_5s: bool,
//           last_multicast_rx_at: float|null, mdns_active: bool, transport: str}
//           Powers the "Local network / multicast" self-test check row.
//           Handled gracefully when absent (404 → row reports "unknown").
//   Existing planner/activate API (omlx/cluster/routes.py, unchanged):
//     POST   /admin/api/cluster/models          — per-node model inventory
//     POST   /admin/api/cluster/catalogue       — model fit across nodes
//     POST   /admin/api/cluster/peer-probe      — SSH reachability / versions / RDMA
//     POST   /admin/api/cluster/autoconfigure   — benchmark probe (measure_performance)
//     POST   /admin/api/cluster/plan            — signed shard plan
//     GET    /admin/api/cluster/deployments     — active deployments
//     POST   /admin/api/cluster/deployments     — activate an approved plan
//     DELETE /admin/api/cluster/deployments/{id}— deactivate
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
        unpair: (nodeId) =>
            `/api/cluster/devices/${encodeURIComponent(nodeId)}`,
        models: '/admin/api/cluster/models',
        catalogue: '/admin/api/cluster/catalogue',
        peerProbe: '/admin/api/cluster/peer-probe',
        autoconfigure: '/admin/api/cluster/autoconfigure',
        plan: '/admin/api/cluster/plan',
        deployments: '/admin/api/cluster/deployments',
        deployment: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}`,
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
        tailscale: { label: 'Tailscale', icon: 'globe' },
        unknown: { label: 'Network', icon: 'help-circle' },
    };

    return {
        // ---- snapshot state -------------------------------------------------
        devicesPayload: null,
        devicesLoaded: false,
        devicesError: '',
        devicesFailureCount: 0,
        devicesUnreachable: false,
        deploymentsPayload: [],
        deploymentsLoaded: false,
        discoveryHealth: null,
        discoveryHealthUnsupported: false,

        // ---- wizard cursor ---------------------------------------------------
        // Null = derive from the snapshot. Explicit values: 'checks', 'plan'.
        stage: null,
        pairing: { target: null, code: '', busy: false, error: '' },
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
        plan: null,
        planLoading: false,
        planError: '',
        activateBusy: false,
        confirmUnpairFor: '',
        confirmDeactivateFor: '',

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

        async refreshDiscoveryHealth() {
            try {
                this.discoveryHealth = await this.apiFetch(
                    CLUSTER_V2_API.discoveryHealth,
                );
                this.discoveryHealthUnsupported = false;
            } catch (error) {
                if (error?.status === 404) {
                    // STUB endpoint not implemented yet — the check row shows
                    // "unknown" instead of a misleading red failure.
                    this.discoveryHealthUnsupported = true;
                    this.discoveryHealth = null;
                } else {
                    this.discoveryHealth = null;
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

        activeDeployment() {
            return this.deploymentsPayload.length
                ? this.deploymentsPayload[0]
                : null;
        },

        // =====================================================================
        // State machine — empty / discovering / device_card / pairing / checks /
        // plan / active / error
        // =====================================================================
        wizardState() {
            if (this.devicesUnreachable) return 'error';
            if (this.activeDeployment()) return 'active';
            if (this.stage === 'plan' && this.pairedDevices().length) {
                return 'plan';
            }
            if (this.stage === 'checks' && this.pairedDevices().length) {
                return 'checks';
            }
            if (this.pairing.target || this.pendingApprovals().length) {
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
            const steps = [
                { key: 'discover', title: 'Find devices', hint: 'Automatic on your network' },
                { key: 'pair', title: 'Pair', hint: 'One code, both Macs' },
                { key: 'checks', title: 'Check', hint: 'SSH · model · RDMA' },
                { key: 'plan', title: 'Split the model', hint: 'Layers per Mac' },
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
            const activeSlot = slotFor[this.wizardState()] ?? 0;
            return steps.map((step, index) => ({
                ...step,
                state:
                    this.wizardState() === 'error'
                        ? 'todo'
                        : index < activeSlot ||
                          (this.wizardState() === 'active' && index === 4)
                        ? 'done'
                        : index === activeSlot
                        ? 'active'
                        : 'todo',
            }));
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
                this.startChecks();
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
            const addrs = Array.isArray(device?.addrs) ? device.addrs : [];
            const first = addrs.find((addr) => addr && addr.ip);
            // Pairing enrollment records the SSH target; until the registry
            // surfaces it, the first verified probe address is the target.
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
            this.checks.benchmarkRunning = true;
            try {
                const result = await this.apiFetch(
                    CLUSTER_V2_API.autoconfigure,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            nodes: this.planNodes(),
                            hosts: this.deploymentHosts(),
                            measure_performance: true,
                            preflight: false,
                            detect_transports: true,
                        }),
                    },
                );
                this.checks.benchmark = { ok: true, result };
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
                        fix: 'On the failing Mac, connect the Thunderbolt cable and verify with `rdma_ctl status` in Terminal, then Re-run checks. Without it the cluster falls back to the slower TCP ring.',
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
                        : 'pending',
                    detail: bench
                        ? bench.ok
                            ? 'Measured compute and link speeds shape the layer split.'
                            : bench.error
                        : 'Optional but recommended — run it once so the split matches real speeds.',
                    fix: 'Benchmark failures are usually a sleeping peer. Wake the other Mac and press Run benchmark again.',
                });
            }

            // 6. Multicast / Local Network self-test (STUB endpoint).
            {
                const health = this.discoveryHealth;
                let status = 'pending';
                let detail = 'Checks whether discovery beacons are arriving.';
                let fix =
                    'macOS is blocking local-network beacons. System Settings → Privacy & Security → Local Network → allow oMLX, then restart oMLX. The check clears itself.';
                if (this.discoveryHealthUnsupported) {
                    status = 'unknown';
                    detail =
                        'This build does not report discovery health yet (endpoint pending). Discovery itself may still work.';
                    fix = '';
                } else if (health) {
                    if (health.multicast_rx_within_5s) {
                        status = 'pass';
                        detail = 'Discovery beacons are flowing on this network.';
                    } else {
                        status = 'fail';
                        detail =
                            'No discovery beacons received in the last 5 seconds.';
                    }
                } else if (this.checks.running) {
                    status = 'running';
                }
                rows.push({
                    key: 'multicast',
                    label: 'Local network permission',
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
            if (!this.modelOptions.length && !this.modelsLoading) {
                this.loadModels();
            }
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
        },

        filteredModels() {
            const query = (this.modelSearch || '').trim().toLowerCase();
            if (!query) return this.modelOptions;
            return this.modelOptions.filter((model) =>
                `${model.id || ''} ${model.model_path || ''}`
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

        shortModelName(model) {
            const path = model?.model_path || '';
            return path.split('/').filter(Boolean).pop() || 'this model';
        },

        modelPresenceLabel(model) {
            const holders = new Set(
                (model?.locations || []).map((loc) => loc.node_id),
            );
            const total = this.allDevices().length || 1;
            return `on ${Math.min(holders.size, total)} of ${total} Macs`;
        },

        async selectModel(model) {
            if (!model || this.planLoading) return;
            this.selectedModelPath = model.model_path;
            await this.runPlan();
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
                    // workstation reserve; everything else is headless.
                    role: 'workstation',
                    memory_guard_tier: 'balanced',
                    accelerator: 'metal',
                });
            }
            for (const peer of this.pairedDevices()) {
                nodes.push({
                    node_id: peer.node_id,
                    capacity_bytes:
                        (this.deviceRamGb(peer) || 0) * 1024 ** 3,
                    role: 'headless',
                    memory_guard_tier: 'balanced',
                    accelerator: 'metal',
                });
            }
            return nodes.filter((node) => node.capacity_bytes > 0);
        },

        async runPlan() {
            if (!this.selectedModelPath) return;
            this.planLoading = true;
            this.planError = '';
            this.plan = null;
            try {
                const model = this.selectedModel();
                const body = {
                    model_path: this.selectedModelPath,
                    nodes: this.planNodes(),
                    execution_profile: 'balanced',
                };
                if (
                    model?.model_source &&
                    model.model_source !== '127.0.0.1'
                ) {
                    body.model_source = model.model_source;
                }
                this.plan = await this.apiFetch(CLUSTER_V2_API.plan, {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
            } catch (error) {
                this.planError =
                    error?.message || 'Could not build the layer split';
            } finally {
                this.planLoading = false;
            }
        },

        planAssignments() {
            const assignments = this.plan?.assignments;
            return Array.isArray(assignments) ? assignments : [];
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
                : assignment.node_id || `Rank ${assignment.rank}`;
        },

        resolvedBackend() {
            const members = [this.selfDevice(), ...this.pairedDevices()].filter(
                Boolean,
            );
            const allJaccl =
                members.length > 1 &&
                members.every((device) => device?.caps?.jaccl);
            return allJaccl ? 'jaccl' : 'ring';
        },

        backendLabel() {
            return this.resolvedBackend() === 'jaccl'
                ? 'JACCL · Thunderbolt RDMA'
                : 'TCP ring';
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

        async activatePlan() {
            if (!this.plan || this.activateBusy) return;
            this.activateBusy = true;
            try {
                const model = this.selectedModel();
                const body = {
                    model_path: this.selectedModelPath,
                    backend: this.resolvedBackend(),
                    nodes: this.planNodes(),
                    hosts: this.deploymentHosts(),
                    approved_placement: this.plan.placement_signature,
                    execution_profile: 'balanced',
                };
                if (
                    model?.model_source &&
                    model.model_source !== '127.0.0.1'
                ) {
                    body.model_source = model.model_source;
                }
                await this.apiFetch(CLUSTER_V2_API.deployments, {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
                this.notify(
                    'success',
                    'Cluster activated. The model starts across your Macs on next load.',
                );
                this.stage = null;
                this.plan = null;
                await this.refreshDeployments();
            } catch (error) {
                if (error?.status === 409) {
                    this.notify(
                        'warning',
                        'The plan changed since you reviewed it — rebuilding it now.',
                    );
                    await this.runPlan();
                } else {
                    this.notify(
                        'error',
                        error?.message || 'Activation failed',
                    );
                }
            } finally {
                this.activateBusy = false;
            }
        },

        async deactivateDeployment(deployment) {
            const id = deployment?.deployment_id;
            if (!id) return;
            if (this.confirmDeactivateFor !== id) {
                this.confirmDeactivateFor = id;
                return;
            }
            this.confirmDeactivateFor = '';
            try {
                await this.apiFetch(CLUSTER_V2_API.deployment(id), {
                    method: 'DELETE',
                });
                this.notify('info', 'Cluster deactivated.');
                await this.refreshDeployments();
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not deactivate',
                );
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
