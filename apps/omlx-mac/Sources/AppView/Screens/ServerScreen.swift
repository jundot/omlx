// PR 7 — real Server screen. ServerHero shows live ServerProcess.State and
// drives Start / Stop / Restart through AppServices. Network + Logging rows
// read/write `/admin/api/global-settings`.
//
// Scope vs design: rows whose backing field doesn't exist server-side yet
// (CORS, HTTPS, Request Timeout, Telemetry, GPU memory, KV-cache quant) are
// deferred until those settings are added to GlobalSettingsRequest. We keep
// the shipped surface honest: every row in this screen is fully wired.

import SwiftUI
import AppKit

struct ServerScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = ServerScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ServerHeroCard(vm: vm)

            SectionHeader("Network")
            ListGroup {
                Row(label: "Listen Address") {
                    Popup(
                        selection: vm.bind($vm.host, save: { vm.saveHost(services: services) }),
                        width: 220,
                        options: [
                            ("127.0.0.1", "127.0.0.1 (Local only)"),
                            ("0.0.0.0", "0.0.0.0 (All networks)"),
                            ("localhost", "localhost"),
                        ]
                    )
                }
                Row(
                    label: "Port",
                    sublabel: "Default 8080. Server restarts on save.",
                    isLast: true
                ) {
                    TextInput(text: $vm.portText, mono: true, width: 90)
                        .onSubmit { vm.savePort(services: services) }
                }
            }

            SectionHeader("API Endpoints")
            APIEndpointsList(host: vm.effectiveHost, port: vm.effectivePort)

            SectionHeader(
                "Default Profile",
                subtitle: "Fallback values used when a model has no profile, or when a profile leaves a field empty"
            )
            ServerDefaultProfileEditor(vm: vm)

            SectionHeader("Logging")
            ListGroup {
                Row(label: "Log Level", isLast: true) {
                    Popup(
                        selection: vm.bind($vm.logLevel, save: vm.saveLogLevel),
                        width: 130,
                        options: [
                            ("error",   "Error"),
                            ("warning", "Warning"),
                            ("info",    "Info"),
                            ("debug",   "Debug"),
                            ("trace",   "Trace"),
                        ]
                    )
                }
            }

            SectionHeader(
                "Storage",
                subtitle: "Where models, settings, logs, and the SSD cache live."
            )
            ListGroup {
                Row(
                    label: "Base Path",
                    sublabel: "OMLX_BASE_PATH. Files move and the server "
                        + "restarts when this changes."
                ) {
                    TextInput(text: $vm.basePathText, mono: true, width: 280)
                }
                Row(
                    label: "Models Directory",
                    sublabel: "Where the server reads and writes model "
                        + "weights. Downloaded models land here.",
                    isLast: true
                ) {
                    TextInput(text: $vm.modelDirText, mono: true, width: 280)
                }
            }
            HStack {
                Spacer()
                Button("Apply") { vm.applyStorage(services: services) }
                    .buttonStyle(.omlx(.primary))
                    .disabled(!vm.hasPendingStorageChanges(services: services)
                              || vm.isMovingBasePath)
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)

            HintFooter(error: vm.lastError)
        }
        .task {
            // services.config is already populated by AppDelegate before this
            // view is mounted, so .onChange never fires for the initial value —
            // mirror it explicitly on first appearance.
            vm.applyConfig(services.config)
            await vm.load(client: services.client)
        }
        .onChange(of: services.config) { _, _ in
            vm.applyConfig(services.config)
        }
        .onChange(of: services.serverState) { _, _ in
            // After a restart triggered by saving host/port, reload to pick
            // up the new effective values.
            Task { await vm.load(client: services.client) }
        }
    }
}

// MARK: - Server hero

/// Hero card shared between the Server and Status screens (omlx-screens.jsx
/// uses the same component on both, lines 75 and 833). When a `vm` is wired
/// in (Server screen), the Restart button folds any pending port/host edits
/// into the restart. On the Status screen there is no such VM, so it just
/// asks AppServices to bounce the cached endpoint.
struct ServerHeroCard: View {
    var vm: ServerScreenVM? = nil

    @EnvironmentObject private var services: AppServices
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 16) {
            Squircle(gradient: SquircleGradient.server, size: 52) {
                Text("oM")
                    .font(.omlxText(22, weight: .heavy))
                    .kerning(-0.5)
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 10) {
                    Text("oMLX Server")
                        .font(.omlxText(18, weight: .semibold))
                        .foregroundStyle(theme.text)
                    StatusPill(status: pillStatus)
                }
                Text(subtitle)
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            }
            Spacer(minLength: 12)
            buttons
        }
        .padding(18)
        .background(heroBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }

    @ViewBuilder
    private var buttons: some View {
        switch services.serverState {
        case .running, .unresponsive:
            HStack(spacing: 6) {
                Button {
                    // Pick up any pending edits in Listen Address / Port that
                    // weren't committed via Enter/blur, so this button is a
                    // single "apply + restart" affordance rather than only
                    // restarting on the cached endpoint. When the hero is
                    // mounted without a VM (Status screen) we simply bounce.
                    if let vm {
                        vm.restart(services: services)
                    } else {
                        Task { try? await services.restartServer() }
                    }
                } label: {
                    Label("Restart", systemImage: "arrow.clockwise")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.normal))

                Button {
                    Task { await services.stopServer() }
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.destructive))
            }

        case .starting, .stopping:
            Button("Working…") { }
                .buttonStyle(.omlx(.normal))
                .disabled(true)

        case .stopped, .failed:
            Button {
                _ = try? services.startServer()
            } label: {
                Label("Start Server", systemImage: "play.fill")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!services.hasServer)
        }
    }

    private var pillStatus: StatusPill.Status {
        switch services.serverState {
        case .running:      return .running
        case .starting:     return .starting
        case .stopping:     return .stopping
        case .stopped:      return .stopped
        case .unresponsive: return .custom(color: theme.amberDot, label: "Unresponsive", fillBg: true)
        case .failed:       return .error
        }
    }

    private var subtitle: String {
        let host = services.config.host
        let port = services.config.port
        switch services.serverState {
        case .running, .unresponsive:
            return "Listening on \(host):\(port)"
        case .starting:
            return "Starting on \(host):\(port)…"
        case .stopping:
            return "Stopping…"
        case .stopped:
            return "Not running"
        case .failed(let m):
            return m
        }
    }

    @ViewBuilder
    private var heroBackground: some View {
        if theme.isDark {
            LinearGradient(
                colors: [
                    Color(rgb24: 0x30D158, opacity: 0.08),
                    Color(rgb24: 0x0A84FF, opacity: 0.06),
                ],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        } else {
            LinearGradient(
                colors: [
                    Color(rgb24: 0xF4FAF6),
                    Color(rgb24: 0xF4F7FC),
                ],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        }
    }
}

// MARK: - Default profile editor

/// Editor for `GlobalSettings.sampling` (server-wide defaults).
///
/// The HTML design surfaces 15 fields here. The server's `SamplingSettings`
/// dataclass currently backs 6 (context, max-tokens, temperature, top_p,
/// top_k, repetition_penalty). The other 9 (min_p, presence_penalty, TTL,
/// thinking, force sampling, pin, etc) are per-model only — we render
/// them disabled with a "Per-model only" tag so the user knows where to
/// look. Expander mirrors the design's "Show all fields…" affordance.
private struct ServerDefaultProfileEditor: View {
    @ObservedObject var vm: ServerScreenVM

    @State private var expanded: Bool = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ListGroup {
                Row(label: "Context Window",
                    sublabel: "Maximum prompt + completion tokens.") {
                    TextInput(text: $vm.samplingContextText, mono: true, suffix: "tk", width: 110)
                        .onSubmit { vm.saveSamplingContext() }
                }
                Row(label: "Max Tokens",
                    sublabel: "Server-wide cap on generated tokens.") {
                    TextInput(text: $vm.samplingMaxTokensText, mono: true, suffix: "tk", width: 110)
                        .onSubmit { vm.saveSamplingMaxTokens() }
                }
                Row(label: "Temperature",
                    sublabel: "Sampling randomness (0–2).") {
                    TextInput(text: $vm.samplingTemperatureText, placeholder: "0.7", mono: true, width: 90)
                        .onSubmit { vm.saveSamplingTemperature() }
                }
                Row(label: "Top P",
                    sublabel: "Nucleus sampling cutoff (0–1).") {
                    TextInput(text: $vm.samplingTopPText, mono: true, width: 90)
                        .onSubmit { vm.saveSamplingTopP() }
                }
                Row(label: "Top K",
                    sublabel: "Limit candidates to top K. 0 = disabled.") {
                    TextInput(text: $vm.samplingTopKText, mono: true, width: 90)
                        .onSubmit { vm.saveSamplingTopK() }
                }
                Row(label: "Repetition Penalty",
                    sublabel: "Penalize repeated tokens.",
                    isLast: !expanded
                ) {
                    TextInput(text: $vm.samplingRepetitionPenaltyText, mono: true, width: 90)
                        .onSubmit { vm.saveSamplingRepetitionPenalty() }
                }
                if expanded {
                    // The remaining design rows aren't server-backed yet —
                    // surfaced disabled with a "Per-model only" pill so the
                    // user knows to set them in the per-model Advanced tab.
                    perModelOnlyRow(label: "Min P", note: "Server defaults don't include min_p; set on a model profile.")
                    perModelOnlyRow(label: "Presence Penalty", note: "Per-model only.")
                    perModelOnlyRow(label: "TTL", note: "Per-model only — see Models → [model] → Basic.")
                    perModelOnlyRow(label: "Enable Thinking", note: "Per-model only — set on a profile.")
                    perModelOnlyRow(label: "Limit Tool Output", note: "Per-model only.")
                    perModelOnlyRow(label: "Force Sampling", note: "Per-model only.")
                    perModelOnlyRow(label: "Pin in memory", note: "Per-model only.")
                    perModelOnlyRow(label: "Speculative decoding", note: "Per-model only — see Models → [model] → Advanced.", isLast: true)
                }
            }
            HStack {
                Spacer()
                Button {
                    expanded.toggle()
                } label: {
                    Text(expanded ? "Show fewer" : "Show all fields…")
                        .font(.omlxText(11.5, weight: .medium))
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .padding(.horizontal, 14)
                .padding(.top, 4)
                .padding(.bottom, 10)
            }
        }
    }

    @ViewBuilder
    private func perModelOnlyRow(label: String, note: String, isLast: Bool = false) -> some View {
        Row(label: label, sublabel: note, isLast: isLast) {
            Text("Per-model only")
                .font(.omlxText(10.5, weight: .heavy))
                .kerning(0.6)
                .textCase(.uppercase)
                .foregroundStyle(theme.textTertiary)
                .padding(.horizontal, 8)
                .frame(height: 22)
                .background(
                    Capsule().fill(theme.codeBg)
                )
                .overlay(
                    Capsule().strokeBorder(theme.inputBorder, lineWidth: 0.5)
                )
        }
    }
}

// MARK: - Endpoints

private struct APIEndpointsList: View {
    let host: String
    let port: Int

    var body: some View {
        ListGroup {
            Row(label: "OpenAI-compatible") {
                CodeChip(value: "http://\(host):\(port)/v1")
            }
            Row(label: "Anthropic / Claude Code") {
                CodeChip(value: "http://\(host):\(port)")
            }
            Row(label: "Health probe") {
                CodeChip(value: "http://\(host):\(port)/health")
            }
            Row(label: "Metrics (Prometheus)", isLast: true) {
                CodeChip(value: "http://\(host):\(port)/metrics")
            }
        }
    }
}

// MARK: - Footer hint

private struct HintFooter: View {
    let error: String?
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(theme.textTertiary)
                Text("Endpoints update live as you change Listen Address and Port. Some changes (host, port) take effect after a server restart.")
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
            if let error {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.redDot)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
    }
}

// MARK: - View model

@MainActor
final class ServerScreenVM: ObservableObject {
    @Published var host: String = "127.0.0.1"
    @Published var portText: String = "8080"
    @Published var logLevel: String = "info"
    @Published var basePathText: String = AppConfig.defaultBasePath()
    @Published var modelDirText: String = ""
    @Published var lastError: String?
    @Published private(set) var isMovingBasePath: Bool = false

    // Server default profile (GlobalSettings.sampling). Backed by 6
    // server-side fields; the other design rows render disabled.
    @Published var samplingContextText: String = "32768"
    @Published var samplingMaxTokensText: String = "32768"
    @Published var samplingTemperatureText: String = "1.0"
    @Published var samplingTopPText: String = "0.95"
    @Published var samplingTopKText: String = "0"
    @Published var samplingRepetitionPenaltyText: String = "1.0"

    /// Last applied (effective) values used to build endpoint URLs. Distinct
    /// from `host`/`portText` so the URLs don't flicker mid-edit.
    @Published var effectiveHost: String = "127.0.0.1"
    @Published var effectivePort: Int = 8080

    private weak var client: OMLXClient?
    private var hasLoaded = false

    func load(client: OMLXClient) async {
        self.client = client
        do {
            let dto = try await client.getGlobalSettings()
            self.host = dto.server.host
            self.portText = String(dto.server.port)
            self.logLevel = canonicalize(level: dto.server.logLevel)
            self.effectiveHost = dto.server.host
            self.effectivePort = dto.server.port
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
        } catch {
            self.lastError = describe(error)
        }
    }

    // MARK: - Default-profile saves

    func saveSamplingContext() {
        guard let n = Int(samplingContextText.trimmingCharacters(in: .whitespaces)), n > 0 else {
            self.lastError = "Context Window must be a positive integer."
            return
        }
        Task { await commit(GlobalSettingsPatch(samplingMaxContextWindow: n)) }
    }

    func saveSamplingMaxTokens() {
        guard let n = Int(samplingMaxTokensText.trimmingCharacters(in: .whitespaces)), n > 0 else {
            self.lastError = "Max Tokens must be a positive integer."
            return
        }
        Task { await commit(GlobalSettingsPatch(samplingMaxTokens: n)) }
    }

    func saveSamplingTemperature() {
        guard let v = Double(samplingTemperatureText.trimmingCharacters(in: .whitespaces)),
              v >= 0, v <= 2 else {
            self.lastError = "Temperature must be a number in [0, 2]."
            return
        }
        Task { await commit(GlobalSettingsPatch(samplingTemperature: v)) }
    }

    func saveSamplingTopP() {
        guard let v = Double(samplingTopPText.trimmingCharacters(in: .whitespaces)),
              v >= 0, v <= 1 else {
            self.lastError = "Top P must be a number in [0, 1]."
            return
        }
        Task { await commit(GlobalSettingsPatch(samplingTopP: v)) }
    }

    func saveSamplingTopK() {
        guard let n = Int(samplingTopKText.trimmingCharacters(in: .whitespaces)), n >= 0 else {
            self.lastError = "Top K must be ≥ 0."
            return
        }
        Task { await commit(GlobalSettingsPatch(samplingTopK: n)) }
    }

    func saveSamplingRepetitionPenalty() {
        guard let v = Double(samplingRepetitionPenaltyText.trimmingCharacters(in: .whitespaces)),
              v >= 0 else {
            self.lastError = "Repetition Penalty must be a non-negative number."
            return
        }
        Task { await commit(GlobalSettingsPatch(samplingRepetitionPenalty: v)) }
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
            self.host = config.host
            self.portText = String(config.port)
            self.effectiveHost = config.host
            self.effectivePort = config.port
        }
        // basePath/modelDir always mirror the live config — they're not
        // gated by `hasLoaded` because the global-settings PATCH path
        // (which sets `hasLoaded = true`) doesn't carry them. modelDir is
        // always a literal path (default `<basePath>/models` or whatever
        // the user pointed it at) — never blank.
        self.basePathText = config.basePath
        self.modelDirText = config.modelDir
    }

    func saveHost(services: AppServices) {
        let next = host
        Task {
            await commit(GlobalSettingsPatch(host: next))
            do {
                try await services.applyServerEndpoint(host: next)
                self.effectiveHost = next
            } catch {
                self.lastError = describe(error)
            }
        }
    }

    func savePort(services: AppServices) {
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)),
              (1...65535).contains(port) else {
            self.lastError = "Port must be a number between 1 and 65535."
            return
        }
        // Patch settings.json on the running server first so its persisted
        // GlobalSettings stays consistent with the next bind, then bounce
        // the parent's ServerProcess on the new port.
        Task {
            await commit(GlobalSettingsPatch(port: port))
            do {
                try await services.applyServerEndpoint(port: port)
                self.effectivePort = port
            } catch {
                self.lastError = describe(error)
            }
        }
    }

    /// True when either Storage text field differs from the current config.
    /// Drives the Apply button's `disabled` state so we don't bounce the
    /// server for an idempotent click.
    func hasPendingStorageChanges(services: AppServices) -> Bool {
        storageDiff(services: services).hasChanges
    }

    /// Apply the Storage text fields. Restart only fires when at least one
    /// of basePath / modelDir has actually changed; matching values are
    /// no-ops and reach the user as a `sameAsCurrent` error rather than a
    /// silent bounce.
    func applyStorage(services: AppServices) {
        let diff = storageDiff(services: services)

        if !diff.baseChanged && !diff.dirChanged {
            self.lastError = "Nothing to apply — both fields match the current config."
            return
        }
        if diff.baseChanged && diff.normalizedBase.isEmpty {
            self.lastError = "Base path cannot be empty."
            return
        }
        if diff.dirChanged && diff.normalizedModelDir.isEmpty {
            self.lastError = "Models Directory cannot be empty."
            return
        }

        isMovingBasePath = true
        Task {
            defer { Task { @MainActor in self.isMovingBasePath = false } }
            do {
                try await services.applyStorageChanges(
                    basePath: diff.baseChanged ? diff.normalizedBase : nil,
                    modelDir: diff.dirChanged ? diff.normalizedModelDir : nil
                )
                // Echo the now-applied values back so the Apply button
                // disables itself.
                self.basePathText = services.config.basePath
                self.modelDirText = services.config.modelDir
                self.lastError = nil
            } catch {
                self.lastError = describe(error)
            }
        }
    }

    /// Computed diff against `services.config`, with tilde expansion + path
    /// normalization. modelDir always carries a literal path (no
    /// "empty == default" magic). Internal so unit tests can drive it.
    struct StorageDiff: Equatable {
        let normalizedBase: String
        let normalizedModelDir: String
        let baseChanged: Bool
        let dirChanged: Bool
        var hasChanges: Bool { baseChanged || dirChanged }
    }

    func storageDiff(services: AppServices) -> StorageDiff {
        let trimmedBase = basePathText.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedBase = ((trimmedBase as NSString).expandingTildeInPath
                              as NSString).standardizingPath
        let currentBase = (services.config.basePath as NSString).standardizingPath
        let baseChanged = !normalizedBase.isEmpty && normalizedBase != currentBase

        let trimmedDir = modelDirText.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedDir = ((trimmedDir as NSString).expandingTildeInPath
                             as NSString).standardizingPath
        let currentDir = (services.config.modelDir as NSString).standardizingPath
        let dirChanged = !normalizedDir.isEmpty && normalizedDir != currentDir

        return StorageDiff(
            normalizedBase: normalizedBase,
            normalizedModelDir: normalizedDir,
            baseChanged: baseChanged,
            dirChanged: dirChanged
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
        let hostChanged = host != effectiveHost

        if portChanged, let p = parsedPort, !(1...65535).contains(p) {
            self.lastError = "Port must be a number between 1 and 65535."
            return
        }
        if portChanged && parsedPort == nil {
            self.lastError = "Port must be a number between 1 and 65535."
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
                    if hostChanged { self.effectiveHost = host }
                } else {
                    try await services.restartServer()
                }
            } catch {
                self.lastError = describe(error)
            }
        }
    }

    func saveLogLevel() {
        Task { await commit(GlobalSettingsPatch(logLevel: logLevel)) }
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

    private func commit(_ patch: GlobalSettingsPatch) async {
        guard let client else { return }
        do {
            _ = try await client.updateGlobalSettings(patch)
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }

    private func canonicalize(level raw: String) -> String {
        switch raw.lowercased() {
        case "warn":   return "warning"
        default:       return raw.lowercased()
        }
    }
}
