import SwiftUI

@MainActor
@Observable
final class PerformanceScreenVM {
    // Scheduler
    var maxConcurrentText: String = "8"
    var embeddingBatchSizeText: String = "32"
    var chunkedPrefill: Bool = false

    // Memory & Lifecycle
    var prefillMemoryGuard: Bool = false
    var memoryGuardTier: String = "balanced"
    var memoryGuardCustomCeilingText: String = ""
    var idleTimeoutText: String = ""
    var modelFallback: Bool = false

    // Cache
    var cacheEnabled: Bool = true
    var hotCacheOnly: Bool = false
    var hotCacheMaxSize: String = ""
    var ssdCacheDir: String = ""
    var ssdCacheMaxSize: String = ""
    var initialCacheBlocksText: String = ""

    // Loaded baselines (everything that drives Apply's enabled state)
    private(set) var loadedMaxConcurrent: Int = 8
    private(set) var loadedEmbeddingBatchSize: Int = 32
    private(set) var loadedChunkedPrefill: Bool = false
    private(set) var loadedPrefillMemoryGuard: Bool = false
    private(set) var loadedMemoryGuardTier: String = "balanced"
    private(set) var loadedMemoryGuardCustomCeilingGb: Double = 0
    private(set) var loadedIdleTimeoutSeconds: Int? = nil
    private(set) var loadedModelFallback: Bool = false
    private(set) var loadedCacheEnabled: Bool = true
    private(set) var loadedHotCacheOnly: Bool = false
    private(set) var loadedHotCacheMaxSize: String = ""
    private(set) var loadedSsdCacheDir: String = ""
    private(set) var loadedSsdCacheMaxSize: String = ""
    private(set) var loadedInitialCacheBlocks: Int? = nil

    private(set) var isSaving: Bool = false
    var lastError: String?

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
            || canonicalHotCacheMaxSize(hotCacheMaxSize) != loadedHotCacheMaxSize
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
                let hotCacheMaxSize = canonicalHotCacheMaxSize(cache.hotCacheMaxSize ?? "")
                self.hotCacheMaxSize = hotCacheMaxSize
                self.loadedHotCacheMaxSize = hotCacheMaxSize
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
        let hcm = canonicalHotCacheMaxSize(hotCacheMaxSize)
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

    private func canonicalHotCacheMaxSize(_ value: String) -> String {
        let normalized = trim(value)
        if normalized.isEmpty || normalized.lowercased() == "auto" {
            return "0"
        }
        return normalized
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
