// Phase 3 — Performance.
//
// One tab for every "how the engine runs" knob. Three sections:
//   • Scheduler — max_concurrent_requests (moved from ServerScreen for
//     scheduler coherence) + chunked_prefill.
//   • Memory & Lifecycle — process / model memory limits, prefill memory
//     guard, server-wide idle timeout, model fallback routing.
//   • Cache — master enable toggle gates a hot-cache toggle + size, a
//     cold-cache directory + size, and an advanced initial-blocks tuning
//     knob (requires restart).
//
// All twelve fields are server-side already (`omlx/admin/routes.py:191-272`)
// — Phase 3 is pure UI. Single Apply button at the bottom, Storage /
// Network pattern: disabled until at least one trimmed draft diverges
// from its loaded value, and only changed fields are sent in the PATCH
// so out-of-band edits to siblings stay intact.

import SwiftUI

struct PerformanceScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = PerformanceScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SchedulerSection(vm: vm)
            MemoryLifecycleSection(vm: vm)
            CacheSection(vm: vm)

            HStack {
                Spacer()
                Button("Apply") {
                    Task { await vm.save(client: services.client) }
                }
                .buttonStyle(.omlx(.primary))
                .disabled(!vm.hasPendingChanges || vm.isSaving)
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
            }
        }
        .task { await vm.load(client: services.client) }
    }
}

// MARK: - Scheduler

private struct SchedulerSection: View {
    @ObservedObject var vm: PerformanceScreenVM

    var body: some View {
        SectionHeader(
            "Scheduler",
            subtitle: "How many requests run at once and how the engine batches them."
        )

        ListGroup {
            Row(
                label: "Max Concurrent Requests",
                sublabel: "Cap on simultaneous /v1 requests."
            ) {
                TextInput(text: $vm.maxConcurrentText, mono: true, width: 90)
            }
            Row(
                label: "Chunked Prefill",
                sublabel: "Split long prompts across scheduler ticks so other requests can interleave.",
                isLast: true
            ) {
                Toggle("", isOn: $vm.chunkedPrefill)
                    .labelsHidden().toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Memory & Lifecycle

private struct MemoryLifecycleSection: View {
    @ObservedObject var vm: PerformanceScreenVM

    var body: some View {
        SectionHeader(
            "Memory & Lifecycle",
            subtitle: "Ceilings + auto-unload behavior. Memory limits accept \"auto\", \"disabled\", \"24GB\", or \"50%\"."
        )

        ListGroup {
            Row(
                label: "Max Process Memory",
                sublabel: "Total resident memory ceiling for the server process."
            ) {
                TextInput(
                    text: $vm.maxProcessMemory,
                    placeholder: "auto",
                    mono: true,
                    width: 140
                )
            }
            Row(
                label: "Max Model Memory",
                sublabel: "Engine pool ceiling. Models above this won't be auto-loaded."
            ) {
                TextInput(
                    text: $vm.maxModelMemory,
                    placeholder: "auto",
                    mono: true,
                    width: 140
                )
            }
            Row(
                label: "Prefill Memory Guard",
                sublabel: "Preflight prefill memory before kicking the engine. Drops requests that would OOM."
            ) {
                Toggle("", isOn: $vm.prefillMemoryGuard)
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(
                label: "Idle Timeout",
                sublabel: "Server-wide auto-unload after N seconds idle. Empty = disabled. Minimum 60."
            ) {
                TextInput(
                    text: $vm.idleTimeoutText,
                    placeholder: "off",
                    mono: true,
                    suffix: "s",
                    width: 110
                )
            }
            Row(
                label: "Model Fallback",
                sublabel: "When the requested model isn't loaded, route to any loaded model instead of 404.",
                isLast: true
            ) {
                Toggle("", isOn: $vm.modelFallback)
                    .labelsHidden().toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Cache

private struct CacheSection: View {
    @ObservedObject var vm: PerformanceScreenVM

    var body: some View {
        SectionHeader(
            "Cache",
            subtitle: "KV cache spillover. The master switch gates everything below."
        )

        ListGroup {
            Row(
                label: "Cache Enabled",
                sublabel: "Master switch for the engine's KV cache subsystem."
            ) {
                Toggle("", isOn: $vm.cacheEnabled)
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(
                label: "Hot Cache Only",
                sublabel: "Skip SSD spillover. Useful on fast machines with abundant RAM."
            ) {
                Toggle("", isOn: $vm.hotCacheOnly)
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(!vm.cacheEnabled)
            }
            Row(
                label: "Hot Cache Size",
                sublabel: "RAM ceiling for hot cache. \"0\" disables, \"8GB\" or \"auto\" accepted."
            ) {
                TextInput(
                    text: $vm.hotCacheMaxSize,
                    placeholder: "auto",
                    mono: true,
                    width: 140
                )
                .disabled(!vm.cacheEnabled)
            }
            Row(
                label: "SSD Cache Directory",
                sublabel: "Where cold-spillover blocks live. Empty = base_path/cache."
            ) {
                TextInput(
                    text: $vm.ssdCacheDir,
                    placeholder: "<base_path>/cache",
                    mono: true,
                    width: 280
                )
                .disabled(!vm.cacheEnabled || vm.hotCacheOnly)
            }
            Row(
                label: "SSD Cache Size",
                sublabel: "Cold-spillover ceiling. \"auto\" = 10% of SSD capacity."
            ) {
                TextInput(
                    text: $vm.ssdCacheMaxSize,
                    placeholder: "auto",
                    mono: true,
                    width: 140
                )
                .disabled(!vm.cacheEnabled || vm.hotCacheOnly)
            }
            Row(
                label: "Initial Cache Blocks",
                sublabel: "Pre-allocated cache blocks at server start. Requires restart to apply.",
                isLast: true
            ) {
                TextInput(
                    text: $vm.initialCacheBlocksText,
                    placeholder: "auto",
                    mono: true,
                    width: 110
                )
                .disabled(!vm.cacheEnabled)
            }
        }
    }
}

// MARK: - View model

@MainActor
final class PerformanceScreenVM: ObservableObject {
    // Scheduler
    @Published var maxConcurrentText: String = "8"
    @Published var chunkedPrefill: Bool = false

    // Memory & Lifecycle
    @Published var maxProcessMemory: String = ""
    @Published var maxModelMemory: String = ""
    @Published var prefillMemoryGuard: Bool = false
    @Published var idleTimeoutText: String = ""
    @Published var modelFallback: Bool = false

    // Cache
    @Published var cacheEnabled: Bool = true
    @Published var hotCacheOnly: Bool = false
    @Published var hotCacheMaxSize: String = ""
    @Published var ssdCacheDir: String = ""
    @Published var ssdCacheMaxSize: String = ""
    @Published var initialCacheBlocksText: String = ""

    // Loaded baselines (everything that drives Apply's enabled state)
    @Published private(set) var loadedMaxConcurrent: Int = 8
    @Published private(set) var loadedChunkedPrefill: Bool = false
    @Published private(set) var loadedMaxProcessMemory: String = ""
    @Published private(set) var loadedMaxModelMemory: String = ""
    @Published private(set) var loadedPrefillMemoryGuard: Bool = false
    @Published private(set) var loadedIdleTimeoutSeconds: Int? = nil
    @Published private(set) var loadedModelFallback: Bool = false
    @Published private(set) var loadedCacheEnabled: Bool = true
    @Published private(set) var loadedHotCacheOnly: Bool = false
    @Published private(set) var loadedHotCacheMaxSize: String = ""
    @Published private(set) var loadedSsdCacheDir: String = ""
    @Published private(set) var loadedSsdCacheMaxSize: String = ""
    @Published private(set) var loadedInitialCacheBlocks: Int? = nil

    @Published private(set) var isSaving: Bool = false
    @Published var lastError: String?

    var hasPendingChanges: Bool {
        parsedMaxConcurrent != loadedMaxConcurrent
            || chunkedPrefill != loadedChunkedPrefill
            || trim(maxProcessMemory) != loadedMaxProcessMemory
            || trim(maxModelMemory) != loadedMaxModelMemory
            || prefillMemoryGuard != loadedPrefillMemoryGuard
            || parsedIdleTimeout != loadedIdleTimeoutSeconds
            || modelFallback != loadedModelFallback
            || cacheEnabled != loadedCacheEnabled
            || hotCacheOnly != loadedHotCacheOnly
            || trim(hotCacheMaxSize) != loadedHotCacheMaxSize
            || trim(ssdCacheDir) != loadedSsdCacheDir
            || trim(ssdCacheMaxSize) != loadedSsdCacheMaxSize
            || parsedInitialCacheBlocks != loadedInitialCacheBlocks
    }

    func load(client: OMLXClient) async {
        do {
            let s = try await client.getGlobalSettings()
            if let sched = s.scheduler {
                self.maxConcurrentText = String(sched.maxConcurrentRequests)
                self.loadedMaxConcurrent = sched.maxConcurrentRequests
                self.chunkedPrefill = sched.chunkedPrefill ?? false
                self.loadedChunkedPrefill = sched.chunkedPrefill ?? false
            }
            if let mem = s.memory {
                self.maxProcessMemory = mem.maxProcessMemory ?? ""
                self.loadedMaxProcessMemory = mem.maxProcessMemory ?? ""
                self.prefillMemoryGuard = mem.prefillMemoryGuard ?? false
                self.loadedPrefillMemoryGuard = mem.prefillMemoryGuard ?? false
            }
            if let model = s.model {
                self.maxModelMemory = model.maxModelMemory ?? ""
                self.loadedMaxModelMemory = model.maxModelMemory ?? ""
                self.modelFallback = model.modelFallback ?? false
                self.loadedModelFallback = model.modelFallback ?? false
            }
            if let idle = s.idleTimeout {
                self.idleTimeoutText = idle.idleTimeoutSeconds.map { String($0) } ?? ""
                self.loadedIdleTimeoutSeconds = idle.idleTimeoutSeconds
            }
            if let cache = s.cache {
                self.cacheEnabled = cache.enabled
                self.loadedCacheEnabled = cache.enabled
                self.hotCacheOnly = cache.hotCacheOnly ?? false
                self.loadedHotCacheOnly = cache.hotCacheOnly ?? false
                self.hotCacheMaxSize = cache.hotCacheMaxSize ?? ""
                self.loadedHotCacheMaxSize = cache.hotCacheMaxSize ?? ""
                self.ssdCacheDir = cache.ssdCacheDir ?? ""
                self.loadedSsdCacheDir = cache.ssdCacheDir ?? ""
                self.ssdCacheMaxSize = cache.ssdCacheMaxSize ?? ""
                self.loadedSsdCacheMaxSize = cache.ssdCacheMaxSize ?? ""
                self.initialCacheBlocksText = cache.initialCacheBlocks.map { String($0) } ?? ""
                self.loadedInitialCacheBlocks = cache.initialCacheBlocks
            }
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    func save(client: OMLXClient) async {
        // Validate first so a bad field's error surfaces without sending a
        // partial patch.
        guard let mc = parsedMaxConcurrent, mc > 0 else {
            self.lastError = "Max Concurrent Requests must be a positive integer."
            return
        }
        // Idle timeout: empty = leave alone (no patch field for null). Non-
        // empty must be a positive integer; server enforces >= 60 itself.
        let idleTrimmed = idleTimeoutText.trimmingCharacters(in: .whitespaces)
        var idleSeconds: Int? = nil
        if !idleTrimmed.isEmpty {
            guard let n = Int(idleTrimmed), n >= 60 else {
                self.lastError = "Idle Timeout must be ≥ 60 seconds (or empty to leave unchanged)."
                return
            }
            idleSeconds = n
        }
        // Initial cache blocks: empty = leave alone, non-empty must parse.
        let initTrimmed = initialCacheBlocksText.trimmingCharacters(in: .whitespaces)
        var initBlocks: Int? = nil
        if !initTrimmed.isEmpty {
            guard let n = Int(initTrimmed), n > 0 else {
                self.lastError = "Initial Cache Blocks must be a positive integer (or empty)."
                return
            }
            initBlocks = n
        }

        var patch = GlobalSettingsPatch()
        // Scheduler
        if mc != loadedMaxConcurrent { patch.maxConcurrentRequests = mc }
        if chunkedPrefill != loadedChunkedPrefill { patch.chunkedPrefill = chunkedPrefill }
        // Memory & lifecycle
        let mpm = trim(maxProcessMemory)
        if mpm != loadedMaxProcessMemory { patch.maxProcessMemory = mpm }
        let mmm = trim(maxModelMemory)
        if mmm != loadedMaxModelMemory { patch.maxModelMemory = mmm }
        if prefillMemoryGuard != loadedPrefillMemoryGuard {
            patch.memoryPrefillMemoryGuard = prefillMemoryGuard
        }
        if idleSeconds != loadedIdleTimeoutSeconds, let s = idleSeconds {
            patch.idleTimeoutSeconds = s
        }
        if modelFallback != loadedModelFallback { patch.modelFallback = modelFallback }
        // Cache
        if cacheEnabled != loadedCacheEnabled { patch.cacheEnabled = cacheEnabled }
        if hotCacheOnly != loadedHotCacheOnly { patch.hotCacheOnly = hotCacheOnly }
        let hcm = trim(hotCacheMaxSize)
        if hcm != loadedHotCacheMaxSize { patch.hotCacheMaxSize = hcm }
        let scd = trim(ssdCacheDir)
        if scd != loadedSsdCacheDir { patch.ssdCacheDir = scd }
        let scm = trim(ssdCacheMaxSize)
        if scm != loadedSsdCacheMaxSize { patch.ssdCacheMaxSize = scm }
        if initBlocks != loadedInitialCacheBlocks, let n = initBlocks {
            patch.initialCacheBlocks = n
        }

        isSaving = true
        defer { isSaving = false }
        do {
            _ = try await client.updateGlobalSettings(patch)
            // Converge baselines on success.
            self.loadedMaxConcurrent = mc
            self.loadedChunkedPrefill = chunkedPrefill
            self.loadedMaxProcessMemory = mpm
            self.loadedMaxModelMemory = mmm
            self.loadedPrefillMemoryGuard = prefillMemoryGuard
            if let s = idleSeconds { self.loadedIdleTimeoutSeconds = s }
            self.loadedModelFallback = modelFallback
            self.loadedCacheEnabled = cacheEnabled
            self.loadedHotCacheOnly = hotCacheOnly
            self.loadedHotCacheMaxSize = hcm
            self.loadedSsdCacheDir = scd
            self.loadedSsdCacheMaxSize = scm
            if let n = initBlocks { self.loadedInitialCacheBlocks = n }
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    // MARK: - Parsing helpers

    private var parsedMaxConcurrent: Int? {
        Int(maxConcurrentText.trimmingCharacters(in: .whitespaces))
    }

    private var parsedIdleTimeout: Int? {
        let t = idleTimeoutText.trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? nil : Int(t)
    }

    private var parsedInitialCacheBlocks: Int? {
        let t = initialCacheBlocksText.trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? nil : Int(t)
    }

    private func trim(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespaces)
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }
}
