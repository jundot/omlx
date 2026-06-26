import SwiftUI

@MainActor
@Observable
final class ServerScreenVM {
    var host: String = "127.0.0.1"
    var portText: String = "8000"
    var logLevel: String = "info"
    var autoStartOnLaunch: Bool = true

    // Phase 4 — Advanced disclosure.
    var sseKeepaliveMode: String = "chunk"
    /// Comma-separated text shown in the input. Parsed to `[String]` on save
    /// so the user can edit incrementally without intermediate trips to the
    /// server. Empty string clears all aliases.
    var serverAliasesText: String = ""
    var basePathText: String = AppConfig.defaultBasePath()
    var modelDirTexts: [String] = [""]
    var hfCacheEnabled: Bool = true
    var lastError: String?
    private(set) var isMovingBasePath: Bool = false

    // Server default profile (GlobalSettings.sampling). Backed by 6
    // server-side fields; the other design rows render disabled.
    var samplingContextText: String = "32768"
    var samplingMaxTokensText: String = "32768"
    var samplingTemperatureText: String = "1.0"
    var samplingTopPText: String = "0.95"
    var samplingTopKText: String = "0"
    var samplingRepetitionPenaltyText: String = "1.0"

    /// Last applied (effective) values used to build endpoint URLs. Distinct
    /// from `host`/`portText` so the URLs don't flicker mid-edit.
    var effectiveHost: String = "127.0.0.1"
    var effectivePort: Int = 8000
    var appliedBindAddress: String = "127.0.0.1"

    /// Apply-button baselines: snapshots of each Apply-managed draft taken
    /// after a successful `load()` or `applyServerSettings()`. The button's
    /// `disabled` state is `drafts == baselines`, so re-snapshotting on
    /// success collapses the page back to "no pending changes".
    ///
    /// Listen Address / Log Level / SSE Keep-Alive Mode auto-apply via the
    /// `bind()` wrapper and don't need baselines here.
    private var baselinePortText: String = "8000"
    private var baselineSamplingContextText: String = "32768"
    private var baselineSamplingMaxTokensText: String = "32768"
    private var baselineSamplingTemperatureText: String = "1.0"
    private var baselineSamplingTopPText: String = "0.95"
    private var baselineSamplingTopKText: String = "0"
    private var baselineSamplingRepetitionPenaltyText: String = "1.0"
    private var baselineServerAliasesText: String = ""
    private var baselineModelDirs: [String] = []
    private var baselineHfCacheEnabled: Bool = true

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var hasLoaded = false

    func load(client: OMLXClient) async {
        self.client = client
        do {
            let dto = try await client.getGlobalSettings()
            self.host = dto.server.host
            self.portText = String(dto.server.port)
            self.logLevel = canonicalize(level: dto.server.logLevel)
            self.autoStartOnLaunch = dto.server.autoStartOnLaunch ?? true
            self.appliedBindAddress = dto.server.host
            self.effectiveHost = AppConfig.connectableHost(for: dto.server.host)
            self.effectivePort = dto.server.port
            self.sseKeepaliveMode = dto.server.sseKeepaliveMode ?? "chunk"
            self.serverAliasesText = dto.server.serverAliases.joined(separator: ", ")
            let modelDirs = Self.cleanedModelDirs(
                dto.model?.modelDirs ?? dto.model?.modelDir.map { [$0] } ?? []
            )
            if !modelDirs.isEmpty {
                self.modelDirTexts = modelDirs
            }
            self.hfCacheEnabled = dto.huggingface?.hfCacheEnabled ?? true
            if let s = dto.sampling {
                self.samplingContextText = String(s.maxContextWindow)
                self.samplingMaxTokensText = String(s.maxTokens)
                self.samplingTemperatureText = trimDouble(s.temperature)
                self.samplingTopPText = trimDouble(s.topP)
                self.samplingTopKText = String(s.topK)
                self.samplingRepetitionPenaltyText = trimDouble(s.repetitionPenalty)
            }
            self.lastError = nil
            self.hasLoaded = true
            self.snapshotApplyBaselines()
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    // MARK: - Apply orchestrator

    /// Snapshot current draft values as the new "applied" baseline. Called
    /// at the end of `load()` and after a successful `applyServerSettings()`.
    private func snapshotApplyBaselines() {
        let t = { (s: String) in s.trimmingCharacters(in: .whitespaces) }
        baselinePortText = t(portText)
        baselineSamplingContextText = t(samplingContextText)
        baselineSamplingMaxTokensText = t(samplingMaxTokensText)
        baselineSamplingTemperatureText = t(samplingTemperatureText)
        baselineSamplingTopPText = t(samplingTopPText)
        baselineSamplingTopKText = t(samplingTopKText)
        baselineSamplingRepetitionPenaltyText = t(samplingRepetitionPenaltyText)
        baselineServerAliasesText = serverAliasesText
        baselineModelDirs = Self.normalizedModelDirs(from: modelDirTexts)
        baselineHfCacheEnabled = hfCacheEnabled
    }

    /// Apply-button gate: true when any Apply-managed draft diverges from
    /// its baseline, or when Storage has uncommitted changes. Listen
    /// Address / Log Level / SSE Keep-Alive Mode auto-apply via `bind()`
    /// and are intentionally excluded from this check.
    func hasPendingServerChanges(services: AppServices) -> Bool {
        let t = { (s: String) in s.trimmingCharacters(in: .whitespaces) }
        if t(portText) != baselinePortText { return true }
        if t(samplingContextText) != baselineSamplingContextText { return true }
        if t(samplingMaxTokensText) != baselineSamplingMaxTokensText { return true }
        if t(samplingTemperatureText) != baselineSamplingTemperatureText { return true }
        if t(samplingTopPText) != baselineSamplingTopPText { return true }
        if t(samplingTopKText) != baselineSamplingTopKText { return true }
        if t(samplingRepetitionPenaltyText) != baselineSamplingRepetitionPenaltyText { return true }
        if parseAliases(serverAliasesText) != parseAliases(baselineServerAliasesText) { return true }
        if hfCacheEnabled != baselineHfCacheEnabled { return true }
        return hasPendingStorageChanges(services: services)
    }

    /// Page-wide Apply. Validates every dirty Apply-managed field upfront
    /// (bail loudly on bad input — no partial patches), bundles the
    /// non-storage changes into a single `GlobalSettingsPatch`, then runs
    /// the storage move flow if needed. A bundled port change rides along
    /// with the storage move's single restart (passed as `port:`); a
    /// port-only change with no storage move goes through
    /// `applyServerEndpoint` instead.
    func applyServerSettings(services: AppServices) {
        let t = { (s: String) in s.trimmingCharacters(in: .whitespaces) }
        var patch = GlobalSettingsPatch()
        var nextPort: Int? = nil

        if t(portText) != baselinePortText {
            guard let p = Int(t(portText)), (1...65535).contains(p) else {
                self.lastError = String(localized: "server.error.port_invalid",
                                        defaultValue: "Port must be a number between 1 and 65535.",
                                        comment: "Server screen error when port value is out of valid range")
                return
            }
            patch.port = p
            nextPort = p
        }
        if t(samplingContextText) != baselineSamplingContextText {
            guard let n = Int(t(samplingContextText)), n > 0 else {
                self.lastError = String(localized: "server.error.context_window_invalid",
                                        defaultValue: "Context Window must be a positive integer.",
                                        comment: "Server screen error when Context Window input is invalid")
                return
            }
            patch.samplingMaxContextWindow = n
        }
        if t(samplingMaxTokensText) != baselineSamplingMaxTokensText {
            guard let n = Int(t(samplingMaxTokensText)), n > 0 else {
                self.lastError = String(localized: "server.error.max_tokens_invalid",
                                        defaultValue: "Max Tokens must be a positive integer.",
                                        comment: "Server screen error when Max Tokens input is invalid")
                return
            }
            patch.samplingMaxTokens = n
        }
        if t(samplingTemperatureText) != baselineSamplingTemperatureText {
            guard let v = Double(t(samplingTemperatureText)), v >= 0, v <= 2 else {
                self.lastError = String(localized: "server.error.temperature_invalid",
                                        defaultValue: "Temperature must be a number in [0, 2].",
                                        comment: "Server screen error when Temperature input is out of range")
                return
            }
            patch.samplingTemperature = v
        }
        if t(samplingTopPText) != baselineSamplingTopPText {
            guard let v = Double(t(samplingTopPText)), v >= 0, v <= 1 else {
                self.lastError = String(localized: "server.error.top_p_invalid",
                                        defaultValue: "Top P must be a number in [0, 1].",
                                        comment: "Server screen error when Top P input is out of range")
                return
            }
            patch.samplingTopP = v
        }
        if t(samplingTopKText) != baselineSamplingTopKText {
            guard let n = Int(t(samplingTopKText)), n >= 0 else {
                self.lastError = String(localized: "server.error.top_k_invalid",
                                        defaultValue: "Top K must be ≥ 0.",
                                        comment: "Server screen error when Top K input is negative or not a number")
                return
            }
            patch.samplingTopK = n
        }
        if t(samplingRepetitionPenaltyText) != baselineSamplingRepetitionPenaltyText {
            guard let v = Double(t(samplingRepetitionPenaltyText)), v >= 0 else {
                self.lastError = String(localized: "server.error.repetition_penalty_invalid",
                                        defaultValue: "Repetition Penalty must be a non-negative number.",
                                        comment: "Server screen error when Repetition Penalty is invalid")
                return
            }
            patch.samplingRepetitionPenalty = v
        }

        let newAliases = parseAliases(serverAliasesText)
        if newAliases != parseAliases(baselineServerAliasesText) {
            patch.serverAliases = newAliases
        }
        if hfCacheEnabled != baselineHfCacheEnabled {
            patch.hfCacheEnabled = hfCacheEnabled
        }

        let diff = storageDiff(services: services)
        if diff.baseChanged && diff.normalizedBase.isEmpty {
            self.lastError = String(localized: "server.error.base_path_empty",
                                    defaultValue: "Base path cannot be empty.",
                                    comment: "Server screen error when Base Path is empty on Apply")
            return
        }
        if diff.modelDirsChanged && diff.normalizedModelDirs.isEmpty {
            self.lastError = String(localized: "server.error.models_dir_empty",
                                    defaultValue: "At least one model directory is required.",
                                    comment: "Server screen error when all model directory fields are empty on Apply")
            return
        }
        if diff.modelDirsChanged && !diff.baseChanged {
            patch.modelDirs = diff.normalizedModelDirs
        }

        let patchHasFields = patch.port != nil
            || patch.samplingMaxContextWindow != nil
            || patch.samplingMaxTokens != nil
            || patch.samplingTemperature != nil
            || patch.samplingTopP != nil
            || patch.samplingTopK != nil
            || patch.samplingRepetitionPenalty != nil
            || patch.serverAliases != nil
            || patch.hfCacheEnabled != nil
            || patch.modelDirs != nil

        if !patchHasFields && !diff.hasChanges {
            self.lastError = String(localized: "server.error.nothing_to_apply",
                                    defaultValue: "Nothing to apply — every field matches the current config.",
                                    comment: "Server screen error when Apply is tapped with no pending changes")
            return
        }

        if diff.baseChanged { isMovingBasePath = true }
        Task {
            defer {
                Task { @MainActor in
                    if self.isMovingBasePath { self.isMovingBasePath = false }
                }
            }
            do {
                let oldBase = services.config.basePath
                if patchHasFields, let client {
                    _ = try await client.updateGlobalSettings(patch)
                }
                if diff.baseChanged {
                    // Hand the bundled port to the storage flow so its single
                    // restart binds the new port. Without this the restart
                    // reuses the cached --port args and silently keeps the old
                    // port even though we just PATCHed the new one.
                    let relocatedModelDirs = diff.modelDirsChanged
                        ? diff.normalizedModelDirs.map {
                            AppServices.relocate(
                                path: $0,
                                oldBase: oldBase,
                                newBase: diff.normalizedBase
                            )
                        }
                        : nil
                    try await services.applyStorageChanges(
                        basePath: diff.baseChanged ? diff.normalizedBase : nil,
                        modelDirs: relocatedModelDirs,
                        port: nextPort
                    )
                    self.basePathText = services.config.basePath
                    self.modelDirTexts = services.config.effectiveModelDirs
                    if let p = nextPort { self.effectivePort = p }
                } else {
                    if let p = nextPort {
                        try await services.applyServerEndpoint(port: p)
                        self.effectivePort = p
                    }
                    if diff.modelDirsChanged {
                        var updated = services.config
                        updated.setModelDirs(diff.normalizedModelDirs)
                        services.updateConfig(updated)
                        self.modelDirTexts = diff.normalizedModelDirs
                    }
                }
                self.lastError = nil
                self.snapshotApplyBaselines()
            } catch {
                self.lastError = error.omlxDescription
            }
        }
    }

    /// Parse the comma-separated aliases text into a dedupe-preserving list.
    /// Used both to build the outbound patch and to diff drafts against the
    /// saved baseline (which is also parsed before compare so reordering /
    /// whitespace-only edits don't false-trigger Apply).
    private func parseAliases(_ text: String) -> [String] {
        let parts = text
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var seen = Set<String>()
        return parts.filter { seen.insert($0).inserted }
    }

    func modelDirText(at index: Int) -> String {
        guard modelDirTexts.indices.contains(index) else { return "" }
        return modelDirTexts[index]
    }

    func setModelDirText(_ value: String, at index: Int) {
        guard modelDirTexts.indices.contains(index) else { return }
        modelDirTexts[index] = value
    }

    func addModelDirectory(_ value: String = "") {
        modelDirTexts.append(value)
    }

    func removeModelDirectory(at index: Int) {
        guard modelDirTexts.indices.contains(index) else { return }
        if modelDirTexts.count == 1 {
            modelDirTexts[0] = ""
        } else {
            modelDirTexts.remove(at: index)
        }
    }

    func browseModelDirectory(at index: Int) {
        guard modelDirTexts.indices.contains(index) else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "server.model_dirs.browse.prompt",
                              defaultValue: "Select",
                              comment: "NSOpenPanel button label for choosing a model directory")
        panel.message = String(localized: "server.model_dirs.browse.message",
                               defaultValue: "Choose a directory containing MLX model folders.",
                               comment: "NSOpenPanel message for choosing a model directory")
        if panel.runModal() == .OK, let url = panel.url {
            modelDirTexts[index] = url.path
        }
    }

    private static func cleanedModelDirs(_ dirs: [String]) -> [String] {
        dirs
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func normalizedPath(_ path: String) -> String {
        ((path.trimmingCharacters(in: .whitespacesAndNewlines) as NSString)
            .expandingTildeInPath as NSString).standardizingPath
    }

    private static func normalizedModelDirs(from dirs: [String]) -> [String] {
        var seen = Set<String>()
        return dirs.compactMap { raw in
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            let normalized = normalizedPath(trimmed)
            return seen.insert(normalized).inserted ? normalized : nil
        }
    }

    /// Format a double for an editable field: `1.0` → `"1.0"`, `0.95` →
    /// `"0.95"`, drops trailing zeros above the first decimal.
    private func trimDouble(_ v: Double) -> String {
        v.truncatingRemainder(dividingBy: 1) == 0
            ? String(format: "%.1f", v)
            : String(v)
    }

    func applyConfig(_ config: AppConfig) {
        if !hasLoaded {
            self.host = config.bindAddress
            self.portText = String(config.port)
            self.autoStartOnLaunch = config.autoStartOnLaunch
            self.appliedBindAddress = config.bindAddress
            self.effectiveHost = config.host
            self.effectivePort = config.port
        }
        // basePath/modelDirs always mirror the live config before the server
        // settings have loaded. After that, `model_dirs` comes from
        // /global-settings so web-admin edits stay visible instead of being
        // collapsed back to AppConfig's primary path on unrelated config
        // notifications.
        self.basePathText = config.basePath
        if !hasLoaded {
            self.modelDirTexts = config.effectiveModelDirs
        }
    }

    func saveHost(services: AppServices) {
        let next = host
        Task {
            await commit(GlobalSettingsPatch(host: next))
            do {
                try await services.applyServerEndpoint(host: next)
                self.appliedBindAddress = next
                self.effectiveHost = AppConfig.connectableHost(for: next)
            } catch {
                self.lastError = error.omlxDescription
            }
        }
    }

    /// True when either Storage text field differs from the current config.
    /// Drives the Apply button's `disabled` state so we don't bounce the
    /// server for an idempotent click.
    func hasPendingStorageChanges(services: AppServices) -> Bool {
        storageDiff(services: services).hasChanges
    }

    /// Computed diff against the last loaded settings, with tilde expansion,
    /// path normalization, and duplicate removal. The primary model directory
    /// is the first entry in `normalizedModelDirs`.
    struct StorageDiff: Equatable {
        let normalizedBase: String
        let normalizedModelDirs: [String]
        let baseChanged: Bool
        let modelDirsChanged: Bool
        var normalizedModelDir: String { normalizedModelDirs.first ?? "" }
        var dirChanged: Bool { modelDirsChanged }
        var hasChanges: Bool { baseChanged || modelDirsChanged }
    }

    func storageDiff(services: AppServices) -> StorageDiff {
        let normalizedBase = Self.normalizedPath(basePathText)
        let currentBase = Self.normalizedPath(services.config.basePath)
        let baseChanged = !normalizedBase.isEmpty && normalizedBase != currentBase

        let normalizedDirs = Self.normalizedModelDirs(from: modelDirTexts)
        let currentDirs = baselineModelDirs.isEmpty
            ? Self.normalizedModelDirs(from: services.config.effectiveModelDirs)
            : baselineModelDirs
        let modelDirsChanged = normalizedDirs != currentDirs

        return StorageDiff(
            normalizedBase: normalizedBase,
            normalizedModelDirs: normalizedDirs,
            baseChanged: baseChanged,
            modelDirsChanged: modelDirsChanged
        )
    }

    /// Restart wired to the hero button. Folds any pending edits in the
    /// Listen Address / Port fields into the restart so the user can either
    /// hit Enter on the field OR just click Restart — both reach the same
    /// place.
    func restart(services: AppServices) {
        let trimmedPort = portText.trimmingCharacters(in: .whitespaces)
        let parsedPort = Int(trimmedPort)
        let portChanged = parsedPort.map { $0 != effectivePort } ?? false
        let hostChanged = host != appliedBindAddress

        if portChanged, let p = parsedPort, !(1...65535).contains(p) {
            self.lastError = String(localized: "server.error.port_invalid",
                                    defaultValue: "Port must be a number between 1 and 65535.",
                                    comment: "Server screen error when port value is out of valid range")
            return
        }
        if portChanged && parsedPort == nil {
            self.lastError = String(localized: "server.error.port_invalid",
                                    defaultValue: "Port must be a number between 1 and 65535.",
                                    comment: "Server screen error when port value is out of valid range")
            return
        }

        Task {
            do {
                if portChanged || hostChanged {
                    if portChanged, let p = parsedPort {
                        await commit(GlobalSettingsPatch(port: p))
                    }
                    if hostChanged {
                        await commit(GlobalSettingsPatch(host: host))
                    }
                    try await services.applyServerEndpoint(
                        host: hostChanged ? host : nil,
                        port: portChanged ? parsedPort : nil
                    )
                    if let p = parsedPort, portChanged { self.effectivePort = p }
                    if hostChanged {
                        self.appliedBindAddress = host
                        self.effectiveHost = AppConfig.connectableHost(for: host)
                    }
                } else {
                    try await services.restartServer()
                }
            } catch {
                self.lastError = error.omlxDescription
            }
        }
    }

    func saveLogLevel() {
        Task { await commit(GlobalSettingsPatch(logLevel: logLevel)) }
    }

    func saveSseKeepaliveMode() {
        Task { await commit(GlobalSettingsPatch(sseKeepaliveMode: sseKeepaliveMode)) }
    }

    func saveAutoStartOnLaunch(services: AppServices) {
        let enabled = autoStartOnLaunch
        Task {
            do {
                switch services.serverState {
                case .running, .unresponsive:
                    if await commit(GlobalSettingsPatch(autoStartOnLaunch: enabled)) {
                        try services.setAutoStartOnLaunch(enabled, persist: false)
                    }
                default:
                    try services.setAutoStartOnLaunch(enabled)
                }
            } catch {
                self.lastError = error.omlxDescription
            }
        }
    }

    /// Build a `Binding` that calls `save` after the value changes. Used for
    /// Popups that have no `onSubmit` hook.
    func bind<T: Equatable>(
        _ binding: Binding<T>,
        save: @escaping () -> Void
    ) -> Binding<T> {
        Binding(
            get: { binding.wrappedValue },
            set: { newValue in
                let changed = binding.wrappedValue != newValue
                binding.wrappedValue = newValue
                if changed { save() }
            }
        )
    }

    @discardableResult
    private func commit(_ patch: GlobalSettingsPatch) async -> Bool {
        guard let client else { return false }
        do {
            _ = try await client.updateGlobalSettings(patch)
            self.lastError = nil
            return true
        } catch {
            self.lastError = error.omlxDescription
            return false
        }
    }


    private func canonicalize(level raw: String) -> String {
        switch raw.lowercased() {
        case "warn":   return "warning"
        default:       return raw.lowercased()
        }
    }
}
