// Phase 3 — Performance.
//
// One tab for every "how the engine runs" knob. Three sections:
//   • Scheduler — max_concurrent_requests (moved from ServerScreen for
//     scheduler coherence), embedding_batch_size, and chunked_prefill.
//   • Memory & Lifecycle — prefill memory guard tier, server-wide idle
//     timeout, model fallback routing.
//   • Cache — master enable toggle gates a hot-cache toggle + size, a
//     cold-cache directory + size, and an advanced initial-blocks tuning
//     knob (requires restart).
//
// All fields are server-side already (`omlx/admin/routes.py:198-235`)
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
                Button(String(localized: "performance.button.apply",
                              defaultValue: "Apply",
                              comment: "Apply button at the bottom of the Performance screen")) {
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
            String(localized: "performance.section.scheduler",
                   defaultValue: "Scheduler",
                   comment: "Section header for the scheduler rows"),
            subtitle: String(localized: "performance.section.scheduler.sub",
                             defaultValue: "How many requests run at once and how the engine batches them.",
                             comment: "Subtitle for the Scheduler section")
        )

        ListGroup {
            Row(
                label: String(localized: "performance.scheduler.max_concurrent",
                              defaultValue: "Max Concurrent Requests",
                              comment: "Row label for max concurrent requests"),
                sublabel: String(localized: "performance.scheduler.max_concurrent.sub",
                                 defaultValue: "Cap on simultaneous /v1 requests.",
                                 comment: "Sublabel for max concurrent requests")
            ) {
                TextInput(text: $vm.maxConcurrentText, mono: true, width: 90)
            }
            Row(
                label: String(localized: "performance.scheduler.embedding_batch_size",
                              defaultValue: "Embedding Batch Size",
                              comment: "Row label for embedding batch size"),
                sublabel: String(localized: "performance.scheduler.embedding_batch_size.sub",
                                 defaultValue: "Max input texts per embedding forward pass.",
                                 comment: "Sublabel for embedding batch size")
            ) {
                TextInput(text: $vm.embeddingBatchSizeText, mono: true, width: 90)
            }
            Row(
                label: String(localized: "performance.scheduler.chunked_prefill",
                              defaultValue: "Chunked Prefill",
                              comment: "Row label for chunked prefill toggle"),
                sublabel: String(localized: "performance.scheduler.chunked_prefill.sub",
                                 defaultValue: "Split long prompts across scheduler ticks so other requests can interleave.",
                                 comment: "Sublabel for chunked prefill toggle"),
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
            String(localized: "performance.section.memory",
                   defaultValue: "Memory & Lifecycle",
                   comment: "Section header for memory and lifecycle settings"),
            subtitle: String(localized: "performance.section.memory.sub",
                             defaultValue: "Memory admission control and auto-unload behavior.",
                             comment: "Subtitle explaining memory and lifecycle settings")
        )

        ListGroup {
            Row(
                label: String(localized: "performance.memory.prefill_guard",
                              defaultValue: "Prefill Memory Guard",
                              comment: "Row label for prefill memory guard toggle"),
                sublabel: String(localized: "performance.memory.prefill_guard.sub",
                                 defaultValue: "Preflight prefill memory before kicking the engine and defer generation scheduling near the ceiling.",
                                 comment: "Sublabel for prefill memory guard")
            ) {
                Toggle("", isOn: $vm.prefillMemoryGuard)
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(
                label: String(localized: "performance.memory.guard_tier",
                              defaultValue: "Memory Guard Tier",
                              comment: "Row label for memory guard tier popup"),
                sublabel: vm.memoryGuardTierDescription
            ) {
                Popup(
                    selection: $vm.memoryGuardTier,
                    width: 150,
                    options: [
                        ("safe",
                         String(localized: "performance.memory.guard_tier.safe",
                                defaultValue: "Safe",
                                comment: "Memory guard tier option: safe")),
                        ("balanced",
                         String(localized: "performance.memory.guard_tier.balanced",
                                defaultValue: "Balanced",
                                comment: "Memory guard tier option: balanced")),
                        ("aggressive",
                         String(localized: "performance.memory.guard_tier.aggressive",
                                defaultValue: "Aggressive",
                                comment: "Memory guard tier option: aggressive")),
                        ("custom",
                         String(localized: "performance.memory.guard_tier.custom",
                                defaultValue: "Custom",
                                comment: "Memory guard tier option: custom")),
                    ]
                )
                .disabled(!vm.prefillMemoryGuard)
            }
            if vm.prefillMemoryGuard && vm.memoryGuardTier == "custom" {
                Row(
                    label: String(localized: "performance.memory.custom_ceiling",
                                  defaultValue: "Custom Ceiling",
                                  comment: "Row label for memory guard custom ceiling"),
                    sublabel: String(localized: "performance.memory.custom_ceiling.sub",
                                     defaultValue: "Fixed process memory ceiling in GB. Used only with the Custom tier.",
                                     comment: "Sublabel for memory guard custom ceiling")
                ) {
                    TextInput(
                        text: $vm.memoryGuardCustomCeilingText,
                        placeholder: String(localized: "performance.memory.custom_ceiling.placeholder",
                                            defaultValue: "GB",
                                            comment: "Placeholder for custom memory guard ceiling"),
                        mono: true,
                        suffix: "GB",
                        width: 110
                    )
                }
            }
            Row(
                label: String(localized: "performance.memory.idle_timeout",
                              defaultValue: "Idle Timeout",
                              comment: "Row label for idle timeout field"),
                sublabel: String(localized: "performance.memory.idle_timeout.sub",
                                 defaultValue: "Server-wide auto-unload after N seconds idle. Empty = disabled. Minimum 60.",
                                 comment: "Sublabel for idle timeout")
            ) {
                TextInput(
                    text: $vm.idleTimeoutText,
                    placeholder: String(localized: "performance.memory.idle_timeout.placeholder",
                                        defaultValue: "off",
                                        comment: "Placeholder text for the idle timeout field when disabled"),
                    mono: true,
                    suffix: "s",
                    width: 110
                )
            }
            Row(
                label: String(localized: "performance.memory.model_fallback",
                              defaultValue: "Model Fallback",
                              comment: "Row label for model fallback toggle"),
                sublabel: String(localized: "performance.memory.model_fallback.sub",
                                 defaultValue: "When the requested model isn't loaded, route to any loaded model instead of 404.",
                                 comment: "Sublabel for model fallback toggle"),
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
            String(localized: "performance.section.cache",
                   defaultValue: "Cache",
                   comment: "Section header for KV cache settings"),
            subtitle: String(localized: "performance.section.cache.sub",
                             defaultValue: "KV cache spillover. The master switch gates everything below.",
                             comment: "Subtitle for the Cache section")
        )

        ListGroup {
            Row(
                label: String(localized: "performance.cache.enabled",
                              defaultValue: "Cache Enabled",
                              comment: "Row label for the master cache enable toggle"),
                sublabel: String(localized: "performance.cache.enabled.sub",
                                 defaultValue: "Master switch for the engine's KV cache subsystem.",
                                 comment: "Sublabel for the master cache enable toggle")
            ) {
                Toggle("", isOn: $vm.cacheEnabled)
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(
                label: String(localized: "performance.cache.hot_only",
                              defaultValue: "Hot Cache Only",
                              comment: "Row label for the hot cache only toggle"),
                sublabel: String(localized: "performance.cache.hot_only.sub",
                                 defaultValue: "Skip SSD spillover. Useful on fast machines with abundant RAM.",
                                 comment: "Sublabel for hot cache only toggle")
            ) {
                Toggle("", isOn: $vm.hotCacheOnly)
                    .labelsHidden().toggleStyle(.switch)
                    .disabled(!vm.cacheEnabled)
            }
            Row(
                label: String(localized: "performance.cache.hot_size",
                              defaultValue: "Hot Cache Size",
                              comment: "Row label for the hot cache size field"),
                sublabel: String(localized: "performance.cache.hot_size.sub",
                                 defaultValue: "RAM ceiling for hot cache. \"0\" disables, \"8GB\" or \"auto\" accepted.",
                                 comment: "Sublabel describing accepted hot cache size values")
            ) {
                TextInput(
                    text: $vm.hotCacheMaxSize,
                    placeholder: String(localized: "performance.memory.placeholder_auto",
                                        defaultValue: "auto",
                                        comment: "Memory field placeholder meaning automatic"),
                    mono: true,
                    width: 140
                )
                .disabled(!vm.cacheEnabled)
            }
            Row(
                label: String(localized: "performance.cache.ssd_dir",
                              defaultValue: "SSD Cache Directory",
                              comment: "Row label for the SSD cache directory field"),
                sublabel: String(localized: "performance.cache.ssd_dir.sub",
                                 defaultValue: "Where cold-spillover blocks live. Empty = base_path/cache.",
                                 comment: "Sublabel for the SSD cache directory")
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
                label: String(localized: "performance.cache.ssd_size",
                              defaultValue: "SSD Cache Size",
                              comment: "Row label for the SSD cache size field"),
                sublabel: String(localized: "performance.cache.ssd_size.sub",
                                 defaultValue: "Cold-spillover ceiling. \"auto\" = 10% of SSD capacity.",
                                 comment: "Sublabel describing accepted SSD cache size values")
            ) {
                TextInput(
                    text: $vm.ssdCacheMaxSize,
                    placeholder: String(localized: "performance.memory.placeholder_auto",
                                        defaultValue: "auto",
                                        comment: "Memory field placeholder meaning automatic"),
                    mono: true,
                    width: 140
                )
                .disabled(!vm.cacheEnabled || vm.hotCacheOnly)
            }
            Row(
                label: String(localized: "performance.cache.initial_blocks",
                              defaultValue: "Initial Cache Blocks",
                              comment: "Row label for the initial cache blocks field"),
                sublabel: String(localized: "performance.cache.initial_blocks.sub",
                                 defaultValue: "Pre-allocated cache blocks at server start. Requires restart to apply.",
                                 comment: "Sublabel for the initial cache blocks field"),
                isLast: true
            ) {
                TextInput(
                    text: $vm.initialCacheBlocksText,
                    placeholder: String(localized: "performance.memory.placeholder_auto",
                                        defaultValue: "auto",
                                        comment: "Memory field placeholder meaning automatic"),
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
    @Published var embeddingBatchSizeText: String = "32"
    @Published var chunkedPrefill: Bool = false

    // Memory & Lifecycle
    @Published var prefillMemoryGuard: Bool = false
    @Published var memoryGuardTier: String = "balanced"
    @Published var memoryGuardCustomCeilingText: String = ""
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
    @Published private(set) var loadedEmbeddingBatchSize: Int = 32
    @Published private(set) var loadedChunkedPrefill: Bool = false
    @Published private(set) var loadedPrefillMemoryGuard: Bool = false
    @Published private(set) var loadedMemoryGuardTier: String = "balanced"
    @Published private(set) var loadedMemoryGuardCustomCeilingGb: Double = 0
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
            || parsedEmbeddingBatchSize != loadedEmbeddingBatchSize
            || chunkedPrefill != loadedChunkedPrefill
            || prefillMemoryGuard != loadedPrefillMemoryGuard
            || canonicalMemoryGuardTier(memoryGuardTier) != loadedMemoryGuardTier
            || parsedMemoryGuardCustomCeiling != loadedMemoryGuardCustomCeilingGb
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
                let embeddingBatchSize = sched.embeddingBatchSize ?? 32
                self.embeddingBatchSizeText = String(embeddingBatchSize)
                self.loadedEmbeddingBatchSize = embeddingBatchSize
                self.chunkedPrefill = sched.chunkedPrefill ?? false
                self.loadedChunkedPrefill = sched.chunkedPrefill ?? false
            }
            if let mem = s.memory {
                self.prefillMemoryGuard = mem.prefillMemoryGuard ?? false
                self.loadedPrefillMemoryGuard = mem.prefillMemoryGuard ?? false
                let tier = canonicalMemoryGuardTier(mem.memoryGuardTier ?? "balanced")
                self.memoryGuardTier = tier
                self.loadedMemoryGuardTier = tier
                let customGb = mem.memoryGuardCustomCeilingGb ?? 0
                self.memoryGuardCustomCeilingText = customGb > 0 ? trimDouble(customGb) : ""
                self.loadedMemoryGuardCustomCeilingGb = customGb
            }
            if let model = s.model {
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
            self.lastError = error.omlxDescription
        }
    }

    func save(client: OMLXClient) async {
        // Validate first so a bad field's error surfaces without sending a
        // partial patch.
        guard let mc = parsedMaxConcurrent, mc > 0 else {
            self.lastError = String(localized: "performance.error.max_concurrent_invalid",
                                    defaultValue: "Max Concurrent Requests must be a positive integer.",
                                    comment: "Performance screen error when max concurrent input is invalid")
            return
        }
        guard let embeddingBatchSize = parsedEmbeddingBatchSize, embeddingBatchSize > 0 else {
            self.lastError = String(localized: "performance.error.embedding_batch_size_invalid",
                                    defaultValue: "Embedding Batch Size must be a positive integer.",
                                    comment: "Performance screen error when embedding batch size input is invalid")
            return
        }
        // Idle timeout: empty = leave alone (no patch field for null). Non-
        // empty must be a positive integer; server enforces >= 60 itself.
        let idleTrimmed = idleTimeoutText.trimmingCharacters(in: .whitespaces)
        var idleSeconds: Int? = nil
        if !idleTrimmed.isEmpty {
            guard let n = Int(idleTrimmed), n >= 60 else {
                self.lastError = String(localized: "performance.error.idle_timeout_invalid",
                                        defaultValue: "Idle Timeout must be ≥ 60 seconds (or empty to leave unchanged).",
                                        comment: "Performance screen error when idle timeout input is below 60 seconds")
                return
            }
            idleSeconds = n
        }
        // Initial cache blocks: empty = leave alone, non-empty must parse.
        let initTrimmed = initialCacheBlocksText.trimmingCharacters(in: .whitespaces)
        var initBlocks: Int? = nil
        if !initTrimmed.isEmpty {
            guard let n = Int(initTrimmed), n > 0 else {
                self.lastError = String(localized: "performance.error.initial_blocks_invalid",
                                        defaultValue: "Initial Cache Blocks must be a positive integer (or empty).",
                                        comment: "Performance screen error when initial cache blocks input is invalid")
                return
            }
            initBlocks = n
        }
        let tier = canonicalMemoryGuardTier(memoryGuardTier)
        let customCeiling = parsedMemoryGuardCustomCeiling
        if prefillMemoryGuard && tier == "custom" && customCeiling <= 0 {
            self.lastError = String(localized: "performance.error.custom_ceiling_invalid",
                                    defaultValue: "Custom Ceiling must be greater than 0 GB.",
                                    comment: "Performance screen error when custom memory guard ceiling is invalid")
            return
        }

        var patch = GlobalSettingsPatch()
        // Scheduler
        if mc != loadedMaxConcurrent { patch.maxConcurrentRequests = mc }
        if embeddingBatchSize != loadedEmbeddingBatchSize {
            patch.embeddingBatchSize = embeddingBatchSize
        }
        if chunkedPrefill != loadedChunkedPrefill { patch.chunkedPrefill = chunkedPrefill }
        // Memory & lifecycle
        if prefillMemoryGuard != loadedPrefillMemoryGuard {
            patch.memoryPrefillMemoryGuard = prefillMemoryGuard
        }
        if tier != loadedMemoryGuardTier {
            patch.memoryGuardTier = tier
        }
        if customCeiling != loadedMemoryGuardCustomCeilingGb {
            patch.memoryGuardCustomCeilingGb = customCeiling
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
            self.loadedEmbeddingBatchSize = embeddingBatchSize
            self.loadedChunkedPrefill = chunkedPrefill
            self.loadedPrefillMemoryGuard = prefillMemoryGuard
            self.loadedMemoryGuardTier = tier
            self.loadedMemoryGuardCustomCeilingGb = customCeiling
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
            self.lastError = error.omlxDescription
        }
    }

    // MARK: - Parsing helpers

    private var parsedMaxConcurrent: Int? {
        Int(maxConcurrentText.trimmingCharacters(in: .whitespaces))
    }

    private var parsedEmbeddingBatchSize: Int? {
        Int(embeddingBatchSizeText.trimmingCharacters(in: .whitespaces))
    }

    private var parsedIdleTimeout: Int? {
        let t = idleTimeoutText.trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? nil : Int(t)
    }

    private var parsedInitialCacheBlocks: Int? {
        let t = initialCacheBlocksText.trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? nil : Int(t)
    }

    private var parsedMemoryGuardCustomCeiling: Double {
        let t = memoryGuardCustomCeilingText.trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? 0 : (Double(t) ?? -1)
    }

    private func trim(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespaces)
    }

    private func trimDouble(_ v: Double) -> String {
        let rounded = (v * 100).rounded() / 100
        if rounded == Double(Int(rounded)) { return String(Int(rounded)) }
        return String(rounded)
    }

    private func canonicalMemoryGuardTier(_ value: String) -> String {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "safe", "balanced", "aggressive", "custom":
            return normalized
        default:
            return "balanced"
        }
    }

    var memoryGuardTierDescription: String {
        switch canonicalMemoryGuardTier(memoryGuardTier) {
        case "safe":
            return String(localized: "performance.memory.guard_tier.safe.sub",
                          defaultValue: "Most conservative. Leaves more headroom before scheduling memory-heavy prefill.",
                          comment: "Description for safe memory guard tier")
        case "aggressive":
            return String(localized: "performance.memory.guard_tier.aggressive.sub",
                          defaultValue: "Uses more memory before throttling. Best when the Mac has ample headroom.",
                          comment: "Description for aggressive memory guard tier")
        case "custom":
            return String(localized: "performance.memory.guard_tier.custom.sub",
                          defaultValue: "Use a fixed GB ceiling instead of the adaptive tier calculation.",
                          comment: "Description for custom memory guard tier")
        default:
            return String(localized: "performance.memory.guard_tier.balanced.sub",
                          defaultValue: "Default tier. Balances throughput with process memory safety.",
                          comment: "Description for balanced memory guard tier")
        }
    }

}
