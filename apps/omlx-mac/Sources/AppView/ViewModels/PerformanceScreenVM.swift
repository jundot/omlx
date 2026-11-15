import AppKit
import Security
import ServiceManagement
import SwiftUI

@MainActor
@Observable
final class PerformanceScreenVM {
    // Scheduler
    var maxConcurrentText: String = "8"
    var embeddingBatchSizeText: String = "32"
    var chunkedPrefill: Bool = false
    var prefillPriority: String = "context"
    var concurrentPriority: String = "balanced"

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
    private(set) var ssdCacheUnavailable: Bool = false
    var ssdCacheMaxSize: String = ""
    var initialCacheBlocksText: String = ""

    // Loaded baselines (everything that drives Apply's enabled state)
    private(set) var loadedMaxConcurrent: Int = 8
    private(set) var loadedEmbeddingBatchSize: Int = 32
    private(set) var loadedChunkedPrefill: Bool = false
    private(set) var loadedPrefillPriority: String = "context"
    private(set) var loadedConcurrentPriority: String = "balanced"
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

    /// D6: the one helper truth. Refreshed on load and after install/apply;
    /// settable so tests can inject the five status states without a disk read.
    var helperStatus: HelperStatus = PrivilegedHelper.currentStatus()
    private(set) var helperInstalling = false

    var hasPendingChanges: Bool {
        parsedMaxConcurrent != loadedMaxConcurrent
            || parsedEmbeddingBatchSize != loadedEmbeddingBatchSize
            || chunkedPrefill != loadedChunkedPrefill
            || prefillPriority != loadedPrefillPriority
            || concurrentPriority != loadedConcurrentPriority
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
            concurrentPriority: concurrentPriority,
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
                self.concurrentPriority = previous.concurrentPriority
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
                self.concurrentPriority = canonicalConcurrentPriority(sched.concurrentPriority)
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
                let concurrentPriority = canonicalConcurrentPriority(sched.concurrentPriority)
                self.concurrentPriority = concurrentPriority
                self.loadedConcurrentPriority = concurrentPriority
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
                self.ssdCacheUnavailable = cache.ssdCacheUnavailable ?? false
                self.ssdCacheMaxSize = cache.ssdCacheMaxSize ?? ""
                self.loadedSsdCacheMaxSize = cache.ssdCacheMaxSize ?? ""
                self.initialCacheBlocksText = cache.initialCacheBlocks.map { String($0) } ?? ""
                self.loadedInitialCacheBlocks = cache.initialCacheBlocks
            }
            self.systemInfo = s.system
            self.refreshHelperStatus()
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    /// Re-read the helper truth from the local ledger plus the latest server
    /// snapshot. Called on load and after any privileged action.
    func refreshHelperStatus() {
        helperStatus = PrivilegedHelper.currentStatus(systemInfo: systemInfo)
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
        if concurrentPriority != loadedConcurrentPriority {
            patch.concurrentPriority = concurrentPriority
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
            self.loadedConcurrentPriority = concurrentPriority
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

    private func canonicalConcurrentPriority(_ value: String?) -> String {
        guard let value, value == "prefill" || value == "decode" else {
            return "balanced"
        }
        return value
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
        guard let mb = wiredLimitMB else { return "" }
        return "sudo sysctl iogpu.wired_limit_mb=\(mb)"
    }

    /// The requested Metal wired limit in MiB, or nil when the server did not
    /// report a request (there is nothing to raise).
    var wiredLimitMB: Int? {
        let requested = Double(systemInfo?.omlxWiredLimitRequestBytes ?? 0)
        guard requested > 0 else { return nil }
        return Int((requested / 1048576.0).rounded(.up))
    }

    /// Raise the kernel Metal wired limit with one standard macOS admin dialog.
    /// The helper is a fixed, validated command — never a shell string built
    /// from UI text — and the app refreshes the settings after it succeeds.
    func applyWiredLimit(client: OMLXClient) async {
        guard let mb = wiredLimitMB else { return }
        do {
            // The persistent helper is installed with one admin prompt; the
            // one-shot AppleScript remains the fallback if SMJobBless cannot
            // install (ad-hoc build, older OS, refused authorization). A
            // cancelled prompt never falls back: it would prompt a second time.
            do {
                try await PrivilegedHelper.setWiredLimit(mb: mb)
            } catch {
                guard PrivilegedHelper.shouldFallBackToAppleScript(error) else {
                    throw error
                }
                try PrivilegedActions.setWiredLimit(mb: mb)
            }
            await load(client: client)
            lastError = String(localized: "performance.memory.wired_limit_applied",
                               defaultValue: "The wired limit was raised. Restart oMLX if it is already serving.",
                               comment: "Confirmation after the admin dialog raised the Metal wired limit")
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: Helper capability (D6)

    var helperPresentation: HelperPresentation {
        PrivilegedHelper.presentation(for: helperStatus)
    }

    /// "Helper: installed (vX)" / "Helper: not installed" — the permanent row.
    var helperStatusLine: String {
        if helperStatus.installed {
            let version = helperStatus.version.map { "v\($0)" }
                ?? String(localized: "performance.helper.unknown_version",
                          defaultValue: "unknown version",
                          comment: "Helper version label when the install ledger has no version")
            return String(localized: "performance.helper.installed_line",
                          defaultValue: "Helper: installed (\(version))",
                          comment: "Permanent Performance row when the privileged helper is installed")
        }
        return String(localized: "performance.helper.not_installed_line",
                      defaultValue: "Helper: not installed",
                      comment: "Permanent Performance row when the privileged helper is absent")
    }

    var helperInstallTitle: String {
        helperPresentation.installAction == .reinstall
            ? String(localized: "performance.helper.reinstall",
                     defaultValue: "Reinstall…",
                     comment: "Button that re-blesses the privileged helper with one admin dialog")
            : String(localized: "performance.helper.install",
                     defaultValue: "Install helper…",
                     comment: "Button that installs the privileged helper with one admin dialog")
    }

    /// "122880 MB of 122880 MB requested" — nil when the server reported no
    /// request (nothing to apply). Independent of the memory-guard warning.
    var wiredLimitStatusLine: String? {
        guard let requested = helperStatus.wiredLimitRequestedMB, requested > 0 else { return nil }
        // Plain digits (no locale grouping) so the English-only UI reads the
        // same on every host: "29696 MB of 122880 MB requested".
        let current = String(helperStatus.wiredLimitCurrentMB ?? 0)
        let target = String(requested)
        return String(localized: "performance.helper.wired_limit_line",
                      defaultValue: "\(current) MB of \(target) MB requested",
                      comment: "Wired-limit row showing the kernel cap against oMLX's request, in MB")
    }

    var wiredLimitApplyVisible: Bool { helperPresentation.applyVisible }

    /// Install or reinstall the helper from the permanent Performance row.
    /// Never conditional on the wired-limit warning. A same-build reinstall is
    /// a no-prompt reuse and says so.
    func installHelper(client: OMLXClient) async {
        guard !helperInstalling else { return }
        helperInstalling = true
        defer { helperInstalling = false }
        do {
            let state = try PrivilegedHelper.ensureInstalled()
            await load(client: client)
            switch state {
            case .reuse:
                lastError = String(localized: "performance.helper.already_installed",
                                   defaultValue: "The system helper is already installed.",
                                   comment: "Confirmation when reinstalling the same helper build needs no prompt")
            case .bless:
                lastError = String(localized: "performance.helper.installed",
                                   defaultValue: "The system helper was installed.",
                                   comment: "Confirmation after the admin dialog installed the helper")
            case .replace:
                lastError = String(localized: "performance.helper.reinstalled",
                                   defaultValue: "The system helper was reinstalled.",
                                   comment: "Confirmation after the admin dialog replaced an older helper")
            }
        } catch {
            lastError = error.localizedDescription
            refreshHelperStatus()
        }
    }

}

// MARK: - Privileged actions

enum PrivilegedActionError: LocalizedError, Equatable {
    case invalidValue
    case cancelled
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .invalidValue:
            return String(localized: "performance.memory.wired_limit_invalid",
                          defaultValue: "The wired-limit value is outside the supported range.",
                          comment: "Error when the requested Metal wired limit is invalid")
        case .cancelled:
            return String(localized: "performance.memory.wired_limit_cancelled",
                          defaultValue: "The administrator prompt was cancelled; the wired limit was not changed.",
                          comment: "Error when the user cancels the admin dialog")
        case .failed(let message):
            let base = String(localized: "performance.memory.wired_limit_failed",
                              defaultValue: "Could not set the wired limit.",
                              comment: "Prefix for a failed privileged wired-limit action")
            return message.isEmpty ? base : "\(base) \(message)"
        }
    }
}

enum PrivilegedActions {
    /// The only command this slice runs. The value is validated as a positive
    /// MiB integer, so the AppleScript shell string cannot be user-controlled.
    static func setWiredLimit(mb: Int) throws {
        let value = try validatedWiredLimitMB(mb)
        let script = "do shell script \"sysctl iogpu.wired_limit_mb=\(value)\" with administrator privileges"
        guard let appleScript = NSAppleScript(source: script) else {
            throw PrivilegedActionError.failed("")
        }
        var error: NSDictionary?
        _ = appleScript.executeAndReturnError(&error)
        guard let error else { return }
        let code = error[NSAppleScript.errorNumber] as? Int ?? 0
        if code == -128 {
            throw PrivilegedActionError.cancelled
        }
        let message = error[NSAppleScript.errorMessage] as? String ?? ""
        throw PrivilegedActionError.failed(message)
    }

    static func validatedWiredLimitMB(_ mb: Int) throws -> Int {
        guard mb > 0, mb <= 1_048_576 else {
            throw PrivilegedActionError.invalidValue
        }
        return mb
    }
}


// MARK: - Helper capability (D6)

/// The one observable truth about the privileged helper. `installed` and
/// `version` come from the on-disk tool plus the app's install ledger; the
/// wired-limit pair comes from the server's system snapshot. Welcome,
/// Performance and the tests all read this same shape, so "is the helper
/// there?" never needs a shell. The helper is a capability, not a side effect
/// of a low wired limit: nothing here is conditional on the warning.
struct HelperStatus: Equatable, Sendable {
    let installed: Bool
    /// The version the app recorded when it blessed the tool; `nil` when the
    /// tool is absent or predates the ledger. The XPC tool exposes no version
    /// query, so the ledger is the authoritative installed version.
    let version: Int?
    let wiredLimitCurrentMB: Int?
    let wiredLimitRequestedMB: Int?

    var installState: PrivilegedHelper.InstallState {
        PrivilegedHelper.installState(
            helperOnDisk: installed,
            recordedVersion: version,
            bundledVersion: PrivilegedHelper.helperBundleVersion
        )
    }

    /// Apply is offered whenever a request exists and the kernel cap differs
    /// from it — including "apply again" after a raise. Equal => hidden, but
    /// the Install/Reinstall affordance stays visible regardless.
    var wiredLimitApplyVisible: Bool {
        guard let requested = wiredLimitRequestedMB, requested > 0 else { return false }
        return wiredLimitCurrentMB != requested
    }
}

/// The button the permanent Helper row shows. It is never hidden: a person
/// must always be able to install or reinstall the capability (D6).
enum HelperInstallAction: Equatable, Sendable {
    case install
    case reinstall
}

struct HelperPresentation: Equatable, Sendable {
    let installAction: HelperInstallAction
    let applyVisible: Bool
    /// Whether invoking install will actually open an admin prompt (bless or
    /// replace) or be a no-prompt reuse. Drives the confirmation text only.
    let willPrompt: Bool
}

/// Outcome of the uninstall's single helper-removal dialog. A cancel or a
/// failure leaves the rest of the uninstall done and is reported, not fatal.
enum HelperRemovalOutcome: Equatable, Sendable {
    case removed
    case cancelled
    case failed(String)
}

// MARK: - Persistent SMJobBless helper

@objc(OMLXHelperProtocol)
protocol OMLXHelperProtocol {
    func setWiredLimit(_ mb: Int, withReply reply: @escaping (Bool, String) -> Void)
}

enum PrivilegedHelper {
    static let helperID = "app.omlx.helper"
    static let installedPath = "/Library/PrivilegedHelperTools/app.omlx.helper"

    /// Must equal `CFBundleVersion` in `apps/omlx-mac/Helper/Info.plist`.
    /// SMJobBless answers "already installed, success" and never replaces a
    /// blessed tool, so the app records the version it installed and removes
    /// the job when the bundled helper changes. The packaging test
    /// `test_privileged_helper_packaging.py` pins this constant to the plist.
    static let helperBundleVersion = 1
    static let installedVersionKey = "omlx.helper.installedVersion"

    enum InstallState: Equatable {
        /// No helper on disk: one admin prompt via SMJobBless.
        case bless
        /// The recorded helper matches this build: no prompt.
        case reuse
        /// A helper from another build is installed: remove, then bless.
        case replace
    }

    /// Pure decision so the upgrade path is testable without an admin prompt.
    static func installState(
        helperOnDisk: Bool,
        recordedVersion: Int?,
        bundledVersion: Int
    ) -> InstallState {
        guard helperOnDisk else { return .bless }
        guard let recordedVersion, recordedVersion == bundledVersion else {
            return .replace
        }
        return .reuse
    }

    /// Pure status builder so the five D6 states are testable without a disk
    /// read or an admin prompt. An absent tool never carries a stale ledger
    /// version.
    static func status(
        helperOnDisk: Bool,
        recordedVersion: Int?,
        wiredLimitCurrentMB: Int?,
        wiredLimitRequestedMB: Int?
    ) -> HelperStatus {
        HelperStatus(
            installed: helperOnDisk,
            version: helperOnDisk ? recordedVersion : nil,
            wiredLimitCurrentMB: wiredLimitCurrentMB,
            wiredLimitRequestedMB: wiredLimitRequestedMB
        )
    }

    /// The real-world reader both screens call. `systemInfo` is the server's
    /// snapshot (nil in Welcome / before load => the wired-limit pair is nil,
    /// the installed/version pair is still local truth).
    static func currentStatus(
        defaults: UserDefaults = .standard,
        systemInfo: GlobalSettingsDTO.SystemInfo? = nil,
        helperOnDisk: Bool? = nil
    ) -> HelperStatus {
        let exists = helperOnDisk
            ?? FileManager.default.fileExists(atPath: installedPath)
        return status(
            helperOnDisk: exists,
            recordedVersion: defaults.object(forKey: installedVersionKey) as? Int,
            wiredLimitCurrentMB: mb(fromBytes: systemInfo?.iogpuWiredLimitBytes),
            wiredLimitRequestedMB: mb(fromBytes: systemInfo?.omlxWiredLimitRequestBytes)
        )
    }

    private static func mb(fromBytes bytes: Int64?) -> Int? {
        guard let bytes, bytes > 0 else { return nil }
        return Int((Double(bytes) / 1_048_576.0).rounded(.up))
    }

    /// The permanent row's decision: an install affordance is always present;
    /// Apply appears only when the kernel cap differs from the request.
    static func presentation(for status: HelperStatus) -> HelperPresentation {
        HelperPresentation(
            installAction: status.installed ? .reinstall : .install,
            applyVisible: status.wiredLimitApplyVisible,
            willPrompt: status.installState != .reuse
        )
    }

    /// Install or reinstall the helper without touching the wired limit — the
    /// Welcome step and the Performance "Install helper…/Reinstall…" button.
    /// Idempotent and versioned: same build => `.reuse` (no prompt), another
    /// build => remove then bless. The bless/remove effects are injectable so
    /// the sequence is testable without an admin dialog. Returns the action
    /// taken so the caller can report "already installed" vs "installed".
    @discardableResult
    static func ensureInstalled(
        defaults: UserDefaults = .standard,
        bundledVersion: Int = helperBundleVersion,
        helperOnDisk: Bool? = nil,
        bless: () throws -> Void = { try install() },
        remove: () throws -> Void = { try uninstall() }
    ) throws -> InstallState {
        let exists = helperOnDisk
            ?? FileManager.default.fileExists(atPath: installedPath)
        let state = installState(
            helperOnDisk: exists,
            recordedVersion: defaults.object(forKey: installedVersionKey) as? Int,
            bundledVersion: bundledVersion
        )
        switch state {
        case .reuse:
            break
        case .bless:
            try bless()
            defaults.set(bundledVersion, forKey: installedVersionKey)
        case .replace:
            // A failed removal is not fatal: bless refuses to overwrite, so
            // the call below then fails rather than silently keeping the old
            // tool. The caller reports the failure.
            try? remove()
            try bless()
            defaults.set(bundledVersion, forKey: installedVersionKey)
        }
        return state
    }

    /// The uninstall's one helper-removal dialog (SMJobRemove bootouts the
    /// daemon and removes the blessed tool). A cancel or failure is reported
    /// and never aborts the rest of the uninstall.
    static func removeForUninstall(
        defaults: UserDefaults = .standard,
        remove: () throws -> Void = { try uninstall() }
    ) -> HelperRemovalOutcome {
        do {
            try remove()
            defaults.removeObject(forKey: installedVersionKey)
            return .removed
        } catch PrivilegedActionError.cancelled {
            return .cancelled
        } catch {
            return .failed(error.localizedDescription)
        }
    }

    /// A cancelled admin prompt or an out-of-range value must never open a
    /// second prompt: the user already answered, and the AppleScript path
    /// validates the same range. Only an installation or XPC failure falls
    /// back to the one-shot AppleScript.
    static func shouldFallBackToAppleScript(_ error: Error) -> Bool {
        if let actionError = error as? PrivilegedActionError,
           actionError == .cancelled || actionError == .invalidValue {
            return false
        }
        return true
    }

    static func setWiredLimit(
        mb: Int,
        defaults: UserDefaults = .standard,
        bundledVersion: Int = helperBundleVersion,
        helperOnDisk: Bool? = nil
    ) async throws {
        let value = try PrivilegedActions.validatedWiredLimitMB(mb)
        // One install path for both "Apply with Admin…" and the standalone
        // Install/Reinstall button: bless/reuse/replace, then set the limit.
        try ensureInstalled(
            defaults: defaults,
            bundledVersion: bundledVersion,
            helperOnDisk: helperOnDisk
        )
        try await call(value)
    }

    private static func install() throws {
        var authorization: AuthorizationRef?
        let status = AuthorizationCreate(nil, nil, [], &authorization)
        guard status == errAuthorizationSuccess, let authorization else {
            throw PrivilegedActionError.failed(
                "Could not create the administrator authorization (\(status))."
            )
        }
        defer { AuthorizationFree(authorization, []) }
        var error: Unmanaged<CFError>?
        let installed = SMJobBless(
            kSMDomainSystemLaunchd,
            helperID as CFString,
            authorization,
            &error
        )
        if !installed {
            let message = error?.takeRetainedValue().localizedDescription
                ?? "SMJobBless refused to install the helper."
            if message.localizedCaseInsensitiveContains("cancel") {
                throw PrivilegedActionError.cancelled
            }
            throw PrivilegedActionError.failed(message)
        }
    }

    private static func uninstall() throws {
        var authorization: AuthorizationRef?
        let status = AuthorizationCreate(nil, nil, [], &authorization)
        guard status == errAuthorizationSuccess, let authorization else {
            throw PrivilegedActionError.failed(
                "Could not create the administrator authorization (\(status))."
            )
        }
        defer { AuthorizationFree(authorization, []) }
        var error: Unmanaged<CFError>?
        let removed = SMJobRemove(
            kSMDomainSystemLaunchd,
            helperID as CFString,
            authorization,
            true,
            &error
        )
        if !removed {
            let message = error?.takeRetainedValue().localizedDescription
                ?? "SMJobRemove could not remove the installed helper."
            throw PrivilegedActionError.failed(message)
        }
    }

    private static func call(_ mb: Int) async throws {
        let connection = NSXPCConnection(
            machServiceName: helperID,
            options: .privileged
        )
        connection.remoteObjectInterface = NSXPCInterface(
            with: OMLXHelperProtocol.self
        )
        connection.resume()
        defer { connection.invalidate() }

        try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, Error>) in
            let gate = HelperReplyGate(continuation)
            guard let proxy = connection.remoteObjectProxyWithErrorHandler({ error in
                gate.finish(.failure(error))
            }) as? OMLXHelperProtocol else {
                gate.finish(.failure(PrivilegedActionError.failed(
                    "The privileged helper did not expose its interface."
                )))
                return
            }
            proxy.setWiredLimit(mb) { ok, message in
                if ok {
                    gate.finish(.success(()))
                } else {
                    gate.finish(.failure(PrivilegedActionError.failed(message)))
                }
            }
        }
    }
}

private final class HelperReplyGate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    func finish(_ result: Result<Void, Error>) {
        lock.lock()
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.resume(with: result)
    }
}
