import SwiftUI

@MainActor
@Observable
final class PerformanceScreenVM {
    // Scheduler
    var maxConcurrentText: String = "8"
    var embeddingBatchSizeText: String = "32"
    var chunkedPrefill: Bool = false
    var prefillPriority: String = "context"

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
    private(set) var loadedPrefillPriority: String = "context"
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
    @ObservationIgnored
    private var restoreBeforeReset: (() -> Void)?
    var showResetNotice = false
    private(set) var isLoading = false
    private(set) var isResetting = false
    var lastError: String?

    /// System memory snapshot from GET /admin/api/global-settings. Drives
    /// the effective-ceiling preview under the Memory Guard Tier rows,
    /// mirroring the web dashboard's breakdown (dashboard.js
    /// `memoryGuardBreakdownHTML` / `memoryGuardShowWiredLimitWarning`).
    var systemInfo: GlobalSettingsDTO.SystemInfo? = nil

    var hasPendingChanges: Bool {
        parsedMaxConcurrent != loadedMaxConcurrent
            || parsedEmbeddingBatchSize != loadedEmbeddingBatchSize
            || chunkedPrefill != loadedChunkedPrefill
            || prefillPriority != loadedPrefillPriority
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

    func resetDefaults(client: OMLXClient) async {
        guard !isLoading, !isResetting, !showResetNotice else { return }
        isResetting = true
        defer { isResetting = false }
        let previous = (
            maxConcurrentText: maxConcurrentText,
            embeddingBatchSizeText: embeddingBatchSizeText,
            chunkedPrefill: chunkedPrefill,
            prefillPriority: prefillPriority,
            prefillMemoryGuard: prefillMemoryGuard,
            memoryGuardTier: memoryGuardTier,
            memoryGuardCustomCeilingText: memoryGuardCustomCeilingText,
            modelFallback: modelFallback,
            idleTimeoutText: idleTimeoutText,
            cacheEnabled: cacheEnabled,
            hotCacheOnly: hotCacheOnly,
            hotCacheMaxSize: hotCacheMaxSize,
            ssdCacheMaxSize: ssdCacheMaxSize,
            initialCacheBlocksText: initialCacheBlocksText,
            lastError: lastError
        )
        do {
            let s = try await client.getGlobalSettingsDefaults()
            restoreBeforeReset = { [weak self] in
                guard let self else { return }
                self.maxConcurrentText = previous.maxConcurrentText
                self.embeddingBatchSizeText = previous.embeddingBatchSizeText
                self.chunkedPrefill = previous.chunkedPrefill
                self.prefillPriority = previous.prefillPriority
                self.prefillMemoryGuard = previous.prefillMemoryGuard
                self.memoryGuardTier = previous.memoryGuardTier
                self.memoryGuardCustomCeilingText = previous.memoryGuardCustomCeilingText
                self.modelFallback = previous.modelFallback
                self.idleTimeoutText = previous.idleTimeoutText
                self.cacheEnabled = previous.cacheEnabled
                self.hotCacheOnly = previous.hotCacheOnly
                self.hotCacheMaxSize = previous.hotCacheMaxSize
                self.ssdCacheMaxSize = previous.ssdCacheMaxSize
                self.initialCacheBlocksText = previous.initialCacheBlocksText
                self.lastError = previous.lastError
            }
            if let sched = s.scheduler {
                self.maxConcurrentText = String(sched.maxConcurrentRequests)
                let embeddingBatchSize = sched.embeddingBatchSize ?? 32
                self.embeddingBatchSizeText = String(embeddingBatchSize)
                self.chunkedPrefill = sched.chunkedPrefill ?? false
                let priority = sched.prefillPriority == "speed" ? "speed" : "context"
                self.prefillPriority = priority
            }
            if let mem = s.memory {
                self.prefillMemoryGuard = mem.prefillMemoryGuard ?? false
                let tier = canonicalMemoryGuardTier(mem.memoryGuardTier ?? "balanced")
                self.memoryGuardTier = tier
                let customGb = mem.memoryGuardCustomCeilingGb ?? 0
                self.memoryGuardCustomCeilingText = customGb > 0 ? trimDouble(customGb) : ""
            }
            if let model = s.model {
                self.modelFallback = model.modelFallback ?? false
            }
            if let idle = s.idleTimeout {
                self.idleTimeoutText = idle.idleTimeoutSeconds.map { String($0) } ?? ""
            }
            if let cache = s.cache {
                self.cacheEnabled = cache.enabled
                self.hotCacheOnly = cache.hotCacheOnly ?? false
                let hotCacheMaxSize = canonicalHotCacheMaxSize(cache.hotCacheMaxSize ?? "")
                self.hotCacheMaxSize = hotCacheMaxSize
                self.ssdCacheMaxSize = cache.ssdCacheMaxSize ?? ""
                self.initialCacheBlocksText = cache.initialCacheBlocks.map { String($0) } ?? ""
            }

            self.lastError = nil
            self.showResetNotice = true
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func cancelReset() {
        restoreBeforeReset?()
        confirmReset()
    }

    func confirmReset() {
        restoreBeforeReset = nil
        showResetNotice = false
    }

    func load(client: OMLXClient) async {
        isLoading = true
        defer { isLoading = false }
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
                let priority = sched.prefillPriority == "speed" ? "speed" : "context"
                self.prefillPriority = priority
                self.loadedPrefillPriority = priority
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
            self.systemInfo = s.system
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
        // Idle timeout: empty or "0" = explicit null (disable), matching the
        // WebUI's "None (disabled)" option. Non-zero must be >= 60; the
        // server enforces the minimum.
        let idleTrimmed = idleTimeoutText.trimmingCharacters(in: .whitespaces)
        var idlePatch: PatchOptionalInt = .null
        if !idleTrimmed.isEmpty {
            guard let n = Int(idleTrimmed) else {
                self.lastError = String(localized: "performance.error.idle_timeout_invalid",
                                        defaultValue: "Idle Timeout must be ≥ 60 seconds (or empty/0 to disable).",
                                        comment: "Performance screen error when idle timeout input is invalid")
                return
            }
            if n == 0 {
                idlePatch = .null
            } else if n >= 60 {
                idlePatch = .value(n)
            } else {
                self.lastError = String(localized: "performance.error.idle_timeout_invalid",
                                        defaultValue: "Idle Timeout must be ≥ 60 seconds (or empty/0 to disable).",
                                        comment: "Performance screen error when idle timeout input is below 60 seconds")
                return
            }
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
        if prefillPriority != loadedPrefillPriority {
            patch.prefillPriority = prefillPriority
        }
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
        // idleTimeout: empty/0 = disable (.null), >= 60 = set (.value).
        // Only send when it actually changed, like the other fields.
        switch idlePatch {
        case .null where loadedIdleTimeoutSeconds != nil:
            patch.idleTimeoutSeconds = .null
        case .value(let n) where n != loadedIdleTimeoutSeconds:
            patch.idleTimeoutSeconds = .value(n)
        default:
            break
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
            self.loadedPrefillPriority = prefillPriority
            self.loadedPrefillMemoryGuard = prefillMemoryGuard
            self.loadedMemoryGuardTier = tier
            self.loadedMemoryGuardCustomCeilingGb = customCeiling
            switch idlePatch {
            case .null:
                self.loadedIdleTimeoutSeconds = nil
            case .value(let s):
                self.loadedIdleTimeoutSeconds = s
            }
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

    var parsedIdleTimeout: Int? {
        let t = idleTimeoutText.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty, let value = Int(t) else { return nil }
        return value == 0 ? nil : value
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
                          defaultValue: "Keeps about 20% of RAM (6-16 GB) free so heavier apps can run alongside oMLX.",
                          comment: "Description for safe memory guard tier")
        case "aggressive":
            return String(localized: "performance.memory.guard_tier.aggressive.sub",
                          defaultValue: "Leaves only 2% of RAM (1.5-4 GB) for macOS and may compress other apps' memory, so oMLX can use nearly all RAM.",
                          comment: "Description for aggressive memory guard tier")
        case "custom":
            return String(localized: "performance.memory.guard_tier.custom.sub",
                          defaultValue: "Use a fixed GB ceiling instead of the adaptive tier calculation.",
                          comment: "Description for custom memory guard tier")
        default:
            return String(localized: "performance.memory.guard_tier.balanced.sub",
                          defaultValue: "Keeps about 8% of RAM (3-8 GB) free for everyday apps while oMLX runs.",
                          comment: "Description for balanced memory guard tier")
        }
    }

    // MARK: - Effective ceiling preview
    //
    // The server computes each tier's ceiling with the enforcer's own math
    // (system.memory_guard_preview), so this screen and the web dashboard
    // only render it. Custom is clamped here because its value is a draft.

    private static let bytesPerGB = 1073741824.0

    /// Follows the tracked draft fields (tier popup, custom ceiling text)
    /// so the preview updates as the user edits, before Apply.
    var memoryGuardBreakdown: String? {
        guard prefillMemoryGuard, let sys = systemInfo else { return nil }
        let tier = canonicalMemoryGuardTier(memoryGuardTier)
        guard let preview = sys.memoryGuardPreview?[tier] else { return nil }

        func gb(_ bytes: Int64?) -> Double { Double(bytes ?? 0) / Self.bytesPerGB }
        func fmt(_ v: Double) -> String { String(format: "%.1f", v) }

        if tier == "custom" {
            let customGB = parsedMemoryGuardCustomCeiling
            guard customGB > 0 else { return nil }
            let limits = [gb(preview.staticBytes), gb(preview.metalCapBytes)].filter { $0 > 0 }
            let ceiling = max(0, ([customGB] + limits).min() ?? 0)
            let metalCapGB = gb(preview.metalCapBytes)
            if metalCapGB > 0 && abs(ceiling - metalCapGB) < 1e-6 && ceiling < customGB {
                return String(localized: "performance.memory.ceiling_preview.custom_kernel",
                              defaultValue: "Custom ceiling \(fmt(customGB)) GB → effective ceiling \(fmt(ceiling)) GB (kernel Metal limit)",
                              comment: "Ceiling preview when the custom memory guard value is clamped by the kernel Metal limit")
            }
            return String(localized: "performance.memory.ceiling_preview.custom",
                          defaultValue: "Custom ceiling \(fmt(customGB)) GB → effective ceiling \(fmt(ceiling)) GB",
                          comment: "Ceiling preview for the custom memory guard tier")
        }

        let free = fmt(gb(preview.freeBytes))
        let inactive = fmt(gb(preview.inactiveBytes))
        let other = fmt(gb(preview.otherAppsBytes))
        let reserve = fmt(gb(preview.reserveBytes))
        let ceiling = fmt(gb(preview.ceilingBytes))
        if preview.binding == "metal_cap" {
            return String(localized: "performance.memory.ceiling_preview.reserve_tier_kernel",
                          defaultValue: "Free \(free) GB + inactive \(inactive) GB + other apps' compressible \(other) GB − reserve \(reserve) GB → effective ceiling \(ceiling) GB (kernel Metal limit)",
                          comment: "Ceiling preview when the reserve tier ceiling is clamped by the kernel Metal limit")
        }
        return String(localized: "performance.memory.ceiling_preview.reserve_tier",
                      defaultValue: "Free \(free) GB + inactive \(inactive) GB + other apps' compressible \(other) GB − reserve \(reserve) GB → ceiling \(ceiling) GB",
                      comment: "Ceiling preview for the reserve memory guard tiers")
    }

    /// Red warning when the effective Metal cap sits below what oMLX asked
    /// Metal for at start — the ceiling users picked can never be reached
    /// until the kernel sysctl is raised.
    var wiredLimitWarningText: String? {
        guard prefillMemoryGuard, let sys = systemInfo else { return nil }
        let kernel = sys.iogpuWiredLimitBytes ?? 0
        let requested = sys.omlxWiredLimitRequestBytes ?? 0
        guard kernel > 0, requested > 0, kernel < requested else { return nil }
        let kernelGB = String(format: "%.1f", Double(kernel) / Self.bytesPerGB)
        return String(localized: "performance.memory.wired_limit_warning",
                      defaultValue: "Metal caps oMLX at \(kernelGB) GB (kernel iogpu.wired_limit_mb). Raise it in Terminal:",
                      comment: "Warning shown when the kernel Metal wired limit is below the memory ceiling oMLX requested")
    }

    /// The sysctl command paired with `wiredLimitWarningText`.
    var wiredLimitCommand: String {
        let requested = Double(systemInfo?.omlxWiredLimitRequestBytes ?? 0)
        let mb = Int((requested / 1048576.0).rounded(.up))
        return "sudo sysctl iogpu.wired_limit_mb=\(mb)"
    }

}
